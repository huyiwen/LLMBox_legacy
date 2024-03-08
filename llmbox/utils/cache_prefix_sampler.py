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

    def __init__(self, data: Sequence[Tuple[str, ...]], batch_size: int, cache_prefix_level: Optional[int] = None):
        self.data = data
        self.batch_size = batch_size
        self.data_idx = None
        """The index of the data that is currently being processed."""
        # print(data, option_nums, len(data), len(option_nums))
        # exit()

        # split data into (src,) and (src, tgt)
        self.total_prefix_num = len(self.data[0]) - 1
        if cache_prefix_level is None:
            self.cache_prefix_level = self.total_prefix_num
        elif cache_prefix_level < 0:
            self.cache_prefix_level = self.total_prefix_num + cache_prefix_level
        else:
            self.cache_prefix_level = cache_prefix_level
        self.joined_data = [[] for _ in range(self.total_prefix_num)]

        self.grouped_prefixes = [[] for _ in range(self.total_prefix_num)]
        """A mapping from `source` text to its corresponded largest index in `self.data`"""

        for s_idx, (*src, _) in enumerate(self.data):
            for p_idx in range(self.total_prefix_num):
                joined_src = "".join(src[:p_idx + 1])
                self.joined_data[p_idx].append(joined_src)
                if s_idx == 0 or joined_src != self.joined_data[p_idx][s_idx - 1]:
                    self.grouped_prefixes[p_idx].append(s_idx)

    def get_cache(self, sources: List[str]) -> Tuple[Optional[SequenceCache], int]:
        """Get cache for a list of sources. Return None if any source is not cached.

        Return:
            cache (`SequenceCache`): The (left padded) cache for the sources.
            prefix_num (`int`): The number of prefixes that are matched in the cache.
        """
        caches = []

        # logger.warning(f"{list(self.cache_trie.items())}")
        last_idx = None
        repeated_times = 1
        batch_size = len(sources)
        for p, src in enumerate(sources):
            prefixes = list(self.cache_trie.prefix(src))
            if len(prefixes) == 0:
                return None, 0
            trie_idx = int(prefixes[-1][0])
            if trie_idx != last_idx or p == batch_size - 1:
                caches.append(self.cache_data[trie_idx].expand_seq(repeated_times))
                repeated_times = 1
            else:
                repeated_times += 1

            last_idx = trie_idx

        return SequenceCache.pad_and_stack(caches), len(prefixes)

    # @profile_every()
    def set_cache(self, src: str, cache: SequenceCache, prefix_num: int):
        trie_idx = self.cache_trie.insert(src)
        self.cache_data.append(cache)

    def batched(self, iterable):
        for i in range(0, len(iterable), self.batch_size):
            yield iterable[i:i + self.batch_size]

    def __iter__(self) -> Iterator[List[int]]:
        self.cache_trie = Trie()
        self.cache_data: List[SequenceCache] = []

        for p_idx in range(self.cache_prefix_level):
            yield from self.batched(self.grouped_prefixes[p_idx])
        yield from self.batched(range(len(self.data)))

    def __repr__(self) -> str:
        return f"CachePrefixSampler(batch_size={self.batch_size})"
