from collections import OrderedDict
from logging import getLogger
from typing import Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from transformers import DynamicCache
from typing_extensions import Self

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
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states.detach(), value_states.detach(), layer_idx)
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

    def get_seq_iter(self) -> Iterator["SequenceCache"]:
        for seq_idx in range(self.get_seq_num()):
            yield self.get_seq_cache(seq_idx)

    def _apply_cache(self, fn) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        applied = [(fn(key), fn(value)) for key, value in zip(self.key_cache, self.value_cache)]
        key_list, value_list = map(list, zip(*applied))
        return key_list, value_list

    def get_seq_cache(self, seq_idx: int) -> "SequenceCache":
        cache = SequenceCache()
        cache.real_seq_length = [self.real_seq_length[seq_idx]]
        if len(self.last_logits) > seq_idx:
            cache.last_logits = [self.last_logits[seq_idx]]
        cache.key_cache, cache.value_cache = self._apply_cache(lambda x: x[seq_idx:seq_idx + 1, ...])
        return cache

    # def expand_seq(self, repeat_times: int) -> "SequenceCache":
    #     assert self.get_seq_num() == 1, "SequenceCache can only repeat sequence when it contains only one sequence"

    #     cache = SequenceCache()
    #     cache.seen_tokens = self.seen_tokens
    #     cache.real_seq_length = self.real_seq_length * repeat_times
    #     for key, value in enumerate(zip(self.key_cache, self.value_cache)):
    #         cache.key_cache.append(key.expand(repeat_times, -1))
    #         cache.value_cache.append(value.expand(repeat_times, -1))
    #     return cache

    @classmethod
    def pad_and_stack(cls, seq_caches: Sequence["SequenceCache"]) -> Self:
        cache = cls()
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

    def get_cache(self, sources: List[str]) -> Optional[SequenceCache]:
        raise NotImplementedError

    def set_cache(self, src: str, cache: SequenceCache):
        raise NotImplementedError


class CachePrefixSampler(Sampler[List[int]], Cacher):

    def __init__(self, data: List[Tuple[str, str]], option_nums: List[int], batch_size: int):
        self.data = data
        self.option_nums = option_nums
        self.batch_size = batch_size
        # print(data, option_nums, len(data), len(option_nums))
        # exit()

        # for caching
        self.cached: OrderedDict[int, SequenceCache] = OrderedDict()
        self.queued_size = 0
        self.data_idx = None
        self.cache_idx = 0

        # split data into (src,) and (src, tgt)
        self.first_options = []
        self.reverse_src = {}
        """A mapping from `source` text to its corresponded largest index in `self.data`"""

        for idx, (src, _) in enumerate(self.data):
            if len(self.first_options) == 0 or src != self.data[self.first_options[-1]][0]:
                self.first_options.append(idx)
            self.reverse_src[src] = idx

    def get_cache(self, sources: List[str]) -> Optional[SequenceCache]:
        """Get cache for a list of sources. Return None if any source is not cached.

        Return:

        """
        caches = [self.cached.get(self.reverse_src[src]) for src in sources]
        if any(c is None for c in caches):
            return None
        logger.debug(f"Get cache: {sources} -> {caches}")
        if len(caches) == 1:
            return caches[0]
        else:
            return SequenceCache.pad_and_stack(caches)

    def set_cache(self, src: str, cache: SequenceCache):
        if self.data_idx is None:
            raise RuntimeError("Cache can only be set during iteration.")

        self.cached[self.reverse_src[src]] = cache
        self.queued_size += self.option_nums[self.cache_idx]
        self.cache_idx += 1
        logger.debug(f"Set cache: {src}")

    def __iter__(self) -> Iterator[List[int]]:
        self.data_idx = 0
        self.cache_idx = 0
        data_len = len(self.data)
        not_cached = self.first_options
        while self.data_idx < data_len:
            # fetch a btach from data queue
            max_spot = min(self.queued_size, self.batch_size)
            if max_spot > 0:
                data_with_cache = list(range(self.data_idx, self.data_idx + max_spot))
                self.queued_size -= max_spot
                self.data_idx += max_spot

                # pop data that is no longer used to save CUDA memory
                for idx in list(self.cached.keys()):
                    if idx < data_with_cache[0]:
                        _c = self.cached.pop(idx)
                        del _c
                    else:
                        break
                logger.debug(f"Yield with cache: {data_with_cache}")
                yield data_with_cache
            else:
                # we need to cache the sources first. update data queue in set_cache
                logger.debug(f"Yield to cache: {not_cached[:self.batch_size]}")
                yield not_cached[:self.batch_size]
                not_cached = not_cached[self.batch_size:]

        # clear cache
        self.cached.clear()

    def __len__(self) -> int:
        tot_len = 0
        for group in range(0, len(self.first_options), self.batch_size):
            tot_len += 1 + sum(self.option_nums[group:group + self.batch_size]) // self.batch_size
        return tot_len

    def __repr__(self) -> str:
        return f"CachePrefixSampler(batch_size={self.batch_size})"
