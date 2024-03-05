from collections import OrderedDict
from logging import getLogger
from typing import Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data.sampler import Sampler
from transformers import DynamicCache
from typing_extensions import Self

_LegacyCache = Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], ...]

logger = getLogger(__name__)


class SequenceCache(DynamicCache):
    """A cache that supports some sequence level operations."""

    def __init__(self) -> None:
        super().__init__()
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

    def strip(self, num_l: int = 0, num_r: int = 0) -> "SequenceCache":
        if num_l + num_r > 0:
            cache = SequenceCache()
            cache.real_seq_length = [l - num_l - num_r for l in self.real_seq_length]
            for key, value in zip(self.key_cache, self.value_cache):
                cache.key_cache.append(key[..., num_l:-num_r, :])
                cache.value_cache.append(value[..., num_l:-num_r, :])
            cache.seen_tokens = self.seen_tokens - num_l - num_r
            return cache
        else:
            return self

    def get_seq_iter(self) -> Iterator["SequenceCache"]:
        for seq_idx in range(self.get_seq_num()):
            yield self.get_seq_cache(seq_idx)

    def get_seq_cache(self, seq_idx: int) -> "SequenceCache":
        cache = SequenceCache()
        cache.real_seq_length = [self.real_seq_length[seq_idx]]
        for layer_idx, (key, value) in enumerate(zip(self.key_cache, self.value_cache)):
            cache.update(key[seq_idx:seq_idx + 1, ...], value[seq_idx:seq_idx + 1, ...], layer_idx)
        return cache

    def expand_seq(self, repeat_times: int) -> "SequenceCache":
        assert self.get_seq_num() == 1, "SequenceCache can only repeat sequence when it contains only one sequence"

        cache = SequenceCache()
        cache.seen_tokens = self.seen_tokens
        cache.real_seq_length = self.real_seq_length * repeat_times
        for key, value in enumerate(zip(self.key_cache, self.value_cache)):
            cache.key_cache.append(key.expand(repeat_times, -1))
            cache.value_cache.append(value.expand(repeat_times, -1))
        return cache

    @classmethod
    def pad_and_stack(cls, seq_caches: Sequence["SequenceCache"]) -> Self:
        cache = cls()
        cache.real_seq_length = [sc.get_seq_length() for sc in seq_caches]
        max_seq_len = max(cache.real_seq_length)
        max_layer_idx = len(seq_caches[0].key_cache)
        kv_shape = seq_caches[0].key_cache[1].shape
        for layer_idx in range(max_layer_idx):
            key_list = []
            value_list = []
            for sc in seq_caches:
                if sc.get_seq_length() < max_seq_len:
                    padding = torch.zeros(
                        kv_shape[:-2] + (max_seq_len - sc.get_seq_length(), kv_shape[-1]),
                        device=sc.key_cache[layer_idx].device,
                        dtype=sc.key_cache[layer_idx].dtype
                    )
                    key_list.append(torch.concatenate([padding, sc.key_cache[layer_idx]], dim=-2))
                    value_list.append(torch.concatenate([padding, sc.value_cache[layer_idx]], dim=-2))
                else:
                    key_list.append(sc.key_cache[layer_idx])
                    value_list.append(sc.value_cache[layer_idx])
            cache.key_cache.append(torch.cat(key_list, dim=0))
            cache.value_cache.append(torch.cat(value_list, dim=0))
        cache.seen_tokens = max_seq_len
        return cache

    def __repr__(self) -> str:
        return f"SequenceCache(real_seq_length={self.real_seq_length})"


class Cacher:
    """A base class that supports caching for a list of sources."""

    def get_cache(self, sources: List[str]) -> Optional[Tuple[torch.Tensor, SequenceCache]]:
        raise NotImplementedError

    def set_cache(self, src: str, last_logit: torch.Tensor, cache: SequenceCache):
        raise NotImplementedError


class CachePrefixSampler(Sampler[List[int]], Cacher):

    def __init__(self, data: List[Tuple[str, str]], option_nums: List[int], batch_size: int):
        self.data = data
        self.option_nums = option_nums
        self.batch_size = batch_size
        # print(data, option_nums, len(data), len(option_nums))
        # exit()

        # for caching
        self.cached: OrderedDict[int, Tuple[torch.Tensor, SequenceCache]] = OrderedDict()
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

    def get_cache(self, sources: List[str]) -> Optional[Tuple[torch.Tensor, SequenceCache]]:
        """Get cache for a list of sources. Return None if any source is not cached.

        Return:

        """
        caches = [self.cached.get(self.reverse_src[src]) for src in sources]
        if any(c is None for c in caches):
            return None
        last_logits, caches = zip(*caches)
        if len(last_logits) == 1:
            last_logits = last_logits[0].unsqueeze(0)
            caches = caches[0]
        else:
            last_logits = torch.stack(last_logits)
            caches = SequenceCache.pad_and_stack(caches)
        logger.debug(f"Get cache: {sources} -> {last_logits.shape}, {caches}")
        return last_logits, caches

    def set_cache(self, src: str, last_logit: torch.Tensor, cache: SequenceCache):
        if self.data_idx is None:
            raise RuntimeError("Cache can only be set during iteration.")

        self.cached[self.reverse_src[src]] = (last_logit, cache)
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
                        self.cached.pop(idx)
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
