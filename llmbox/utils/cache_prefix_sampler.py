from collections import OrderedDict, defaultdict
from logging import getLogger
from pprint import pformat
from typing import Iterator, List, Optional, Sequence, Tuple

import torch
from cyac import Trie
from pytorch_memlab import profile, profile_every
from torch.utils.data.sampler import Sampler
from transformers import DynamicCache

_LegacyCache = Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], ...]

logger = getLogger(__name__)


class SequenceCache(DynamicCache):
    """A cache that supports some sequence level operations."""

    def __init__(self) -> None:
        # keeps cache in a list instead of a stacked tensor because the tensor may on different devices
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        self.seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.last_logits: List[torch.Tensor] = []
        self.real_seq_length: List[int] = []

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[_LegacyCache] = None) -> "SequenceCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = SequenceCache()
        if past_key_values is not None:
            for key_states, value_states in past_key_values:
                cache.key_cache.append(key_states.detach())
                cache.value_cache.append(value_states.detach())
            cache.real_seq_length = [past_key_values[0][0].shape[2]] * past_key_values[0][0].shape[0]
        return cache

    def get_seq_num(self) -> int:
        return len(self.real_seq_length)

    def remove_paddings(self, num_l: int = 0, num_r: int = 0):
        if num_l + num_r > 0:
            self.real_seq_length = [l - num_l - num_r for l in self.real_seq_length]
            for layer_idx in range(len(self.key_cache)):
                self.key_cache[layer_idx] = self.key_cache[layer_idx][..., num_l:-num_r, :]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][..., num_l:-num_r, :]
            self.seen_tokens = self.seen_tokens - num_l - num_r

    def split_by_seq(self) -> List["SequenceCache"]:
        results = []
        for seq_idx in range(self.get_seq_num()):
            cache = SequenceCache()
            cache.real_seq_length = [self.real_seq_length[seq_idx]]
            if len(self.last_logits) > seq_idx:
                cache.last_logits = [self.last_logits[seq_idx]]
            cache.key_cache, cache.value_cache = self._apply_cache(lambda x: x[seq_idx:seq_idx + 1, ...].clone())
            results.append(cache)
        return results

    def _apply_cache(self, fn) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        applied = [(fn(key), fn(value)) for key, value in zip(self.key_cache, self.value_cache)]
        key_list, value_list = map(list, zip(*applied))
        return key_list, value_list

    def expand_seq(self, repeat_times: int) -> "SequenceCache":
        assert self.get_seq_num() == 1, "SequenceCache can only repeat sequence when it contains only one sequence"

        cache = SequenceCache()
        cache.seen_tokens = self.seen_tokens
        cache.last_logits = self.last_logits * repeat_times
        cache.real_seq_length = self.real_seq_length * repeat_times
        for key, value in zip(self.key_cache, self.value_cache):
            cache.key_cache.append(key.expand(repeat_times, -1, -1, -1))
            cache.value_cache.append(value.expand(repeat_times, -1, -1, -1))
        return cache

    @classmethod
    def pad_and_stack(cls, seq_caches: Sequence["SequenceCache"]) -> "SequenceCache":
        if len(seq_caches) == 1:
            return seq_caches[0]

        cache = SequenceCache()
        for sc in seq_caches:
            cache.last_logits.extend(sc.last_logits)
            cache.real_seq_length.extend(sc.real_seq_length)
        max_seq_len = max(cache.real_seq_length)
        max_layer_idx = len(seq_caches[0].key_cache)
        cache.seen_tokens = max_seq_len

        for layer_idx in range(max_layer_idx):
            key_list = []
            value_list = []
            for sc in seq_caches:
                kv_shape = sc.key_cache[0].shape
                if sc.get_seq_length() < max_seq_len:
                    padding = torch.zeros(
                        kv_shape[:-2] + (max_seq_len - sc.get_seq_length(), kv_shape[-1]),
                        device=sc.key_cache[layer_idx].device,
                        dtype=sc.key_cache[layer_idx].dtype
                    )
                    key_list.append(torch.cat((padding, sc.key_cache[layer_idx]), dim=-2))
                    value_list.append(torch.cat((padding, sc.value_cache[layer_idx]), dim=-2))
                else:
                    key_list.append(sc.key_cache[layer_idx])
                    value_list.append(sc.value_cache[layer_idx])
            cache.key_cache.append(torch.cat(key_list, dim=0))
            cache.value_cache.append(torch.cat(value_list, dim=0))
        return cache

    def __repr__(self) -> str:
        return f"SequenceCache(real_seq_length={self.real_seq_length})"


class Cacher:
    """A base class that supports caching for a list of sources."""

    def get_cache(self, sources: List[str]) -> Tuple[Optional[SequenceCache], int]:
        raise NotImplementedError

    def set_cache(self, src: str, cache: SequenceCache, prefix_num: int):
        raise NotImplementedError


class CachePrefixSampler(Sampler[List[int]], Cacher):
    """A sampler that facilitates key-value caching for a list of text segments."""

    def __init__(
        self,
        data: Sequence[Tuple[str, ...]],
        batch_size: int,
        cache_prefix_level: Optional[int] = None,
        cache_batch_size: Optional[int] = None,
    ):
        self.data = data
        self.batch_size = batch_size
        self.cache_batch_size = cache_batch_size if cache_batch_size else batch_size
        self.data_idx = None
        """The index of the data that is currently being processed."""
        # print(data, option_nums, len(data), len(option_nums))
        # exit()

        # split data into (src,) and (src, tgt)
        if cache_prefix_level is None:
            cache_prefix_level = len(self.data[0]) - 1
        elif cache_prefix_level < 0:
            cache_prefix_level = len(self.data[0]) + cache_prefix_level - 1
        else:
            cache_prefix_level = min(cache_prefix_level, len(self.data[0]) - 1)

        self.total_prefix_num = cache_prefix_level
        self.joined_data = [[] for _ in range(self.total_prefix_num)]
        self.postfix_nums = [[] for _ in range(self.total_prefix_num)]
        self.postfix_nums_2 = defaultdict(int)
        self.cache_range = defaultdict(lambda: [-1, -1])
        """A mapping from `source` text to its corresponded largest index in `self.data`"""

        for s_idx, (*src, _) in enumerate(self.data):
            for p_idx in range(self.total_prefix_num):
                joined_src = "".join(src[:p_idx + 1])
                self.joined_data[p_idx].append(joined_src)
                last_s = self.cache_range[joined_src][1] if joined_src in self.cache_range else None
                if last_s is None or joined_src != self.joined_data[p_idx][last_s]:
                    self.postfix_nums[p_idx].append(1)
                    self.cache_range[joined_src][0] = s_idx  # start
                else:
                    self.postfix_nums[p_idx][-1] += 1
                self.cache_range[joined_src][1] = s_idx + 1  # end
                self.postfix_nums_2[joined_src] += 1

    def get_cache(self, sources: List[str]) -> Tuple[Optional[SequenceCache], int]:
        """Get cache for a list of sources. Return None if any source is not cached.

        Return:
            cache (`SequenceCache`): The (left padded) cache for the sources.
            prefix_num (`int`): The number of prefixes that are matched in the cache.
        """
        caches = []
        prefix_num = -1
        for src in sources:
            results = list(self.cache_trie.prefix(src))
            # check all sources have the same prefix number
            if prefix_num != -1 and len(results) != prefix_num:
                raise RuntimeError(f"Inconsistent prefix number {len(results)} != {prefix_num}\n{src}")
            prefix_num = len(results)

            if prefix_num > 0:
                trie_idx = int(results[-1][0])
                # logger.warning(f"G{trie_idx}")
                caches.append(self.cache_data[trie_idx])
            else:
                return None, 0

        if prefix_num is None:
            raise RuntimeError("No prefix number")
        # logger.warning(f"Get cache: {sources}, {prefix_num}")
        return SequenceCache.pad_and_stack(caches), prefix_num

    def set_cache(self, src: str, cache: SequenceCache, prefix_num: int):
        if self.data_idx is None:
            raise RuntimeError("Cache can only be set during iteration.")

        trie_idx = self.cache_trie.insert(src)
        # logger.warning(f"S{trie_idx}")
        if trie_idx > len(self.cache_data):
            self.cache_data.extend([None] * (trie_idx - len(self.cache_data) + 1))
            self.cache_data.append(cache)
        elif trie_idx == len(self.cache_data):
            self.cache_data.append(cache)
        else:
            self.cache_data[trie_idx] = cache

        self.queued_size[prefix_num] += self.postfix_nums_2[src]
        self.cache_idx[prefix_num] += 1

    def fetch_to_cache(self, data_idx: int, yield_with_cache: bool) -> Tuple[List[int], bool]:
        to_cache = []
        with_cache = []
        last_prefix = None
        cached_idx = len(list(self.cache_trie.prefix(self.joined_data[self.total_prefix_num - 1][data_idx])))
        if yield_with_cache:
            need_cache_idx = self.total_prefix_num - 1
        else:
            need_cache_idx = cached_idx

        while len(to_cache) < self.cache_batch_size and data_idx < self.data_len:
            joined_prefix = self.joined_data[need_cache_idx][data_idx]
            cache_num = len(list(self.cache_trie.prefix(joined_prefix)))
            if joined_prefix != last_prefix and cache_num == need_cache_idx:
                if yield_with_cache:
                    if len(with_cache) > 0:
                        # logger.warning(f"Yield with cache 1: {with_cache}")
                        return with_cache, True
                    else:
                        need_cache_idx = cached_idx
                    yield_with_cache = False
                to_cache.append(data_idx)

            elif yield_with_cache and cache_num == self.total_prefix_num:
                with_cache.append(data_idx)
                # logger.warning(f"Add: {len(joined_prefix)}")
                if len(with_cache) == self.batch_size:
                    # logger.warning(f"Yield with cache 2: {with_cache}")
                    return with_cache, True

            data_idx += 1
            last_prefix = joined_prefix
        # logger.warning(
        #     f"Yield to cache 1: {to_cache} {with_cache} {len(to_cache)} < {self.cache_batch_size} {data_idx} < {self.data_len}"
        # )
        if yield_with_cache:
            return with_cache, True
        else:
            return to_cache, False

    def __iter__(self) -> Iterator[List[int]]:
        self.data_idx = 0
        self.data_len = len(self.data)

        self.cache_trie = Trie()
        self.cache_data: List[SequenceCache] = []
        self.queued_size = [0] * self.total_prefix_num
        self.cache_idx = [0] * self.total_prefix_num

        while self.data_idx < self.data_len:
            assert self.data_idx is not None, "Cache can only be set during iteration."

            if 0 not in self.queued_size:
                to_yield, with_cache = self.fetch_to_cache(self.data_idx, True)
                if with_cache:
                    max_spot = len(to_yield)
                    for idx in range(self.total_prefix_num):
                        self.queued_size[idx] -= max_spot
                    self.data_idx += max_spot

                yield to_yield
            else:
                to_yield, _ = self.fetch_to_cache(self.data_idx, False)
                yield to_yield

            # pop data that is no longer used to save CUDA memory
            for src, idx in self.cache_trie.items():
                _, ed = self.cache_range[src]
                if ed <= to_yield[0]:
                    self.cache_trie.remove(src)
                    self.cache_data[idx] = None

        # clear cache
        self.data_idx = None

    def __repr__(self) -> str:
        return f"CachePrefixSampler(batch_size={self.batch_size}, cache_batch_size={self.cache_batch_size}, total_prefix_num={self.total_prefix_num})"
