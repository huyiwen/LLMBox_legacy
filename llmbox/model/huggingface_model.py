from logging import getLogger
from pprint import pformat, pprint
from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple, Union

import torch
from pytorch_memlab import LineProfiler, profile, profile_every
from torch.nn import CrossEntropyLoss
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from ..utils.cache_prefix_sampler import SequenceCache
from .model import Model

if TYPE_CHECKING:
    from ..utils import ModelArguments

logger = getLogger(__name__)

_Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


def load_hf_model(args: "ModelArguments",
                  return_postfix_tokenizer: bool = True) -> Tuple[PreTrainedModel, _Tokenizer, Optional[_Tokenizer]]:
    logger.info(f"Loading {args.model_name_or_path} using Hugging Face Transformers...")

    model_kwargs = dict(
        torch_dtype=torch.float16,
        device_map=args.device_map,
    )

    if args.prefix_caching:
        model_kwargs["is_decoder"] = True

    if args.flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs).eval()
    except (TypeError, ImportError, ValueError) as e:
        if "attn_implementation" in str(e) or "flash att" in str(e).lower().replace("_", " "):
            logger.warning(
                f"Cannot set `attn_implementation` for {args.model_name_or_path}: {e}. Set `flash_attention` to False."
            )
            args.flash_attention = False
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs).eval()
        else:
            raise e

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path, use_fast=True, padding_side="left", truncation_side="left", add_eos_token=False
    )
    if return_postfix_tokenizer:
        postfix_tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path,
            use_fast=True,
            padding_side="right",
            truncation_side="right",
            add_eos_token=False,
            add_bos_token=False,
        )
        if postfix_tokenizer.pad_token is None:
            postfix_tokenizer.pad_token = postfix_tokenizer.unk_token
    else:
        postfix_tokenizer = None

    # TODO: [Important]!!! check for each tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    # TODO: [Important]!!! check for each tokenizer
    max_length = min(getattr(tokenizer, "tokenizer_model_max_length", 1e10), getattr(args, "max_length") or 1e10)
    for key in [
        "max_sequence_length",
        "max_position_embeddings",
        "model_max_length",
        "seq_length",
        "seq_len",
        "n_positions",
        "max_seq_len",
        "max_seq_length",
    ]:
        max_length = min(max_length, getattr(model.config, key, 1e10))
    if not max_length or max_length >= 1e10:
        max_length = 2048
        logger.warning(
            f"Cannot specify model's maximum length according to `args` or model config. Set to 2048 by default."
        )

    tokenizer.model_max_length = max_length
    logger.debug(f"Model: {model}\nTokenizer: {tokenizer}")
    return model, tokenizer, postfix_tokenizer


class HuggingFaceModel(Model):

    model: PreTrainedModel
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

    def __init__(self, args: "ModelArguments"):
        super().__init__(args)
        self.args = args
        self.type = args.model_type
        if self.type not in {"base", "instruction"}:
            raise ValueError(
                f"Invalid model type: {self.type}. Please use `--model_type` to specify the"
                " model type, which can be chosen from `base` and `instruction`."
            )

        self.model, self.tokenizer, self.postfix_tokenizer = load_hf_model(args, args.prefix_caching)
        self.space_token_id = -100
        if self.postfix_tokenizer is not None:
            input_ids = self.postfix_tokenizer(" postfix").input_ids
            if len(input_ids) > 1:
                self.space_token_id = input_ids[0]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _subsentences_start_idx(offset_mapping: torch.Tensor) -> Iterator[int]:
        r"""Given offset mapping, return the index of the first token in the encoded sentence of each subsentence. The length of the encoded sentence will be yielded at the end, to ensure that the end index of the last subsentence will be included."""
        for token_idx, (char_st, char_ed) in enumerate(offset_mapping):
            if char_st == 0:
                yield token_idx
        yield len(offset_mapping)

    # @profile_every(enable=False)
    def get_cache(
        self,
        batched_inputs: List[str],
        prefix_cache: Optional[SequenceCache] = None,
        return_caches: bool = True,
        save_last_logits: bool = False,
    ) -> Union[List[SequenceCache], Tuple[torch.Tensor, torch.Tensor, List[int]]]:
        """
        Return:
            `return_caches` is True:
                caches (`List[SequenceCache]`): A list of caches for each prefix and input pair without padding. At the same device as `self.device`.
            `return_caches` is False:
                logits (`torch.Tensor`): Logits of batched inputs. At the same device as `self.device`.
                input_ids (`torch.Tensor`): A tensor of input_ids of batched inputs. At the same device as `self.device`.
                input_lengths (`List[int]`): The number of non-padding tokens in each input.
        """
        batch_size = len(batched_inputs)
        if prefix_cache is not None:
            cache_num = prefix_cache.get_seq_num()
            if cache_num != batch_size and cache_num != 1:
                raise RuntimeError(
                    f"The number of sentence in prefix_cache should be one or be equal to the batch size {batch_size}"
                )
            _tokenizer = self.postfix_tokenizer
        else:
            _tokenizer = self.tokenizer

        # keeps the arguments the same as get_ppl, except for arguments specified at the tokenizer class level
        batched_encodings = _tokenizer(
            batched_inputs,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )  # type: ignore
        input_ids = batched_encodings["input_ids"]
        attention_mask = batched_encodings["attention_mask"]
        # logger.info(f"Space {self.space_token_id}")
        if input_ids[:, :1].eq(self.space_token_id).all():
            input_ids = input_ids[:, 1:]
            attention_mask = attention_mask[:, 1:]
        input_lengths = [len(am.nonzero()) for am in attention_mask]
        max_input_len = max(input_lengths)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # logger.info(f"{batched_inputs} {input_ids} {attention_mask} {input_lengths}")
        if prefix_cache is not None:
            max_prefix_len = prefix_cache.get_seq_num()
            prefix_mask = torch.ones((batch_size, prefix_cache.get_seq_length()), device=self.device)
            if cache_num == 1 and batch_size > 1:
                prefix_cache = prefix_cache.expand_seq(batch_size)
                prefix_lengths = [prefix_cache.get_seq_length()] * batch_size
                input_pos = torch.arange(max_prefix_len, max_input_len + max_prefix_len,
                                         device=self.device).expand(batch_size, -1)
            else:
                prefix_lengths = []
                input_pos = []
                for seq_idx in range(batch_size):
                    prefix_len = prefix_cache.real_seq_length[seq_idx]
                    prefix_mask[seq_idx, :-prefix_len] = 0
                    prefix_lengths.append(prefix_len)
                    input_pos.append(torch.arange(prefix_len, max_input_len + prefix_len))
                input_pos = torch.stack(input_pos).to(self.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # type: ignore
            prefix_cache = prefix_cache.to_legacy_cache()  # type: ignore
        else:
            max_prefix_len = 0
            prefix_lengths = [0] * batch_size
            input_pos = None

        with torch.no_grad():
            results = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=input_pos,
                past_key_values=prefix_cache,
                use_cache=True,
            )
            logits = results.logits.detach()

        if return_caches:
            # store the non-padding parts of caches to ensure the correct creation of position_ids when using
            # these caches in the future
            caches = SequenceCache.from_legacy_cache(results.past_key_values).split_by_seq()
            for idx, seq_cache in enumerate(caches):
                seq_cache.remove_paddings(
                    num_l=max_prefix_len - prefix_lengths[idx],
                    num_r=max_input_len - input_lengths[idx],
                )
                if save_last_logits:
                    seq_cache.last_logits = [logits[idx:idx + 1, -1:, :].clone()]
            return caches
        else:
            return logits, input_ids, input_lengths

    def get_ppl_with_cache(self, batched_targets: List[str], prefix_cache: SequenceCache) -> List[Tuple[float, int]]:
        logits, labels, input_lengths = self.get_cache(batched_targets, prefix_cache, return_caches=False)
        last_logits = torch.cat(prefix_cache.last_logits).to(logits.device)
        shift_logits = torch.cat([last_logits, logits[:, :-1]], dim=-2)
        labels[labels == self.tokenizer.pad_token_id] = -100
        probs = self.loss_fct(shift_logits.view(-1, self.model.config.vocab_size),
                              labels.view(-1)).view(labels.size(0), -1).cpu()

        ppls = []
        for prob, tgt_len in zip(probs, input_lengths):
            ppl = sum(prob.tolist()[:tgt_len])
            ppls.append((ppl, tgt_len))
        return ppls

    def set_ppl_args(self, **extra_model_args):
        r"""Set the configurations for PPL score calculation. This is useful because different datasets may have different requirements for ppl calculation."""
        self.loss_fct = CrossEntropyLoss(reduction="none")
        self._ppl_args_set = True
        if len(extra_model_args) > 0:
            logger.warning(f"Unused generation arguments: {extra_model_args}")

    def get_ppl(self, batched_inputs: List[Tuple[str, ...]]) -> List[Tuple[float, int]]:
        if not self._ppl_args_set:
            logger.warning(f"Please set the get_ppl arguments using `set_ppl_args` before calling `get_ppl`.")

        if self.cacher is not None:
            *prefix_groups, targets = list(map(list, zip(*batched_inputs)))
            batch_num = len(prefix_groups[0])

            # if cache is available, get_ppl_with_cache
            all_prefix = ["".join(pg[i] for pg in prefix_groups) for i in range(batch_num)]
            prefix_cache, cached_num = self.cacher.get_cache(all_prefix)
            if prefix_cache is not None and cached_num == len(prefix_groups):
                return self.get_ppl_with_cache(targets, prefix_cache)

            # pass the input without prefix text to the model
            concat_cached_prefix = ["".join(pg[i] for pg in prefix_groups[:cached_num + 1]) for i in range(batch_num)]
            prefix_cache = self.get_cache(
                prefix_groups[cached_num], prefix_cache, save_last_logits=cached_num == len(prefix_groups) - 1
            )

            for p, c in zip(concat_cached_prefix, prefix_cache):
                self.cacher.set_cache(p, c, cached_num)
            return []

        prompt = [src + tgt for src, tgt in batched_inputs]

        batched_encodings = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(
                input_ids=batched_encodings["input_ids"], attention_mask=batched_encodings["attention_mask"]
            ).logits
            shift_logits = logits.detach()[:, :-1].contiguous()
            shift_labels = batched_encodings["input_ids"][:, 1:].contiguous()
            shift_labels[shift_labels == self.tokenizer.pad_token_id] = -100
            probs = self.loss_fct(shift_logits.view(-1, self.model.config.vocab_size),
                                  shift_labels.view(-1)).view(shift_labels.size(0), -1).cpu()
        # logger.info(f"Getppl {shift_logits}, {probs}")

        ppls = []
        for prob, (src, _), offset, attention_mask in zip(
            probs, batched_inputs, batched_encodings.offset_mapping, batched_encodings.attention_mask
        ):
            ppl = [None] + prob.tolist()
            offset = [st for st, ed in offset]
            tgt_start = max(
                # the start index of the last token in source
                offset.index(len(src)),
                # the second nonzero token (the first nonzero token is " " if source is empty)
                attention_mask.nonzero()[0][0].item() + 1,
            )
            tgt_end = len(offset)
            # logger.info(ppl[tgt_start:])
            ppl = sum(ppl[tgt_start:])
            ppls.append((ppl, tgt_end - tgt_start))
        return ppls

    def set_generation_args(self, **extra_model_args):
        generation_kwargs = {}
        for key in [
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "best_of",
            "repetition_penalty",
            "length_penalty",
            "early_stopping",
            "no_repeat_ngram_size",
        ]:
            # ModelArguments > extra_model_args
            value = getattr(self.args, key, None)
            if value is None:
                value = extra_model_args.pop(key, None)
            if key == "max_tokens" and value is None:
                value = 1024
            if value is not None:
                if key == "max_tokens":
                    generation_kwargs["max_new_tokens"] = value
                elif key == "best_of":
                    generation_kwargs["num_beams"] = value
                elif key == "temperature":
                    if value > 0:
                        generation_kwargs["temperature"] = value
                        generation_kwargs["do_sample"] = True
                    else:
                        generation_kwargs["do_sample"] = False
                else:
                    generation_kwargs[key] = value

        generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        self.generation_kwargs = generation_kwargs

        self._generation_args_set = True
        if len(extra_model_args) > 0:
            logger.warning(f"Unused generation arguments: {extra_model_args}")

    def generation(self, batched_inputs: List[str]) -> List[str]:
        """Generate the response of given question for this batch.

        Returns:
            List[str]: The list of generation results.
        """
        if not self._generation_args_set:
            logger.warning(
                f"Please set the generation arguments using `set_generation_args` before calling `generation`."
            )

        batched_encodings = self.tokenizer(
            batched_inputs,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)

        batch_outputs = self.model.generate(**batched_encodings, **self.generation_kwargs)
        max_input_length = batched_encodings["input_ids"].size(1)
        batch_outputs = batch_outputs[:, max_input_length:]
        answers = self.tokenizer.batch_decode(
            batch_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return answers

    def set_prob_args(self, **extra_model_args):
        self._token_labels = []
        self._word_labels = []
        self._candidate_ids = extra_model_args.pop("candidate_ids", None)

        self._prob_args_set = True
        if len(extra_model_args) > 0:
            logger.warning(f"Unused generation arguments: {extra_model_args}")

    def _get_label_ids(self, option_num: Optional[int]) -> List[int]:
        """Return the tokenized labels of options."""
        if option_num is not None:
            if len(self._token_labels) < option_num:
                labels = [chr(i + 65) for i in range(len(self._token_labels), option_num)]
                self._word_labels.extend([self.tokenizer.encode(l, add_special_tokens=False)[0] for l in labels])
                self._token_labels.extend([self.tokenizer.convert_tokens_to_ids(l) for l in labels])
            return self._word_labels[:option_num] + self._token_labels[:option_num]
        else:
            if self._candidate_ids is None:
                raise ValueError("The candidate_ids must be provided when option_num is None.")
            return self._candidate_ids

    def get_prob_with_cache(
        self,
        batched_inputs: List[Tuple[str, int]],
        batched_option_nums: List[int],
        prefix_cache: SequenceCache,
    ) -> List[List[float]]:
        logits, _, _ = self.get_cache(batched_inputs, prefix_cache, return_caches=False)
        batch_logits = logits.detach()[:, -1, :]

        answers = []
        for i, option_num in enumerate(batched_option_nums):
            label_ids = self._get_label_ids(option_num)
            answers.append(torch.softmax(batch_logits[i, label_ids], dim=-1, dtype=torch.float32).tolist())
        return answers

    def get_prob(self, batched_inputs: List[Tuple[str, int]]) -> List[List[float]]:
        if not self._prob_args_set:
            logger.warning(f"Please set the get_prob arguments using `set_prob_args` before calling `get_prob`.")

        if self.cacher is not None:
            *batched_prompts, batched_option_nums = map(list, zip(*batched_inputs))
            batch_num = len(batched_prompts[0])

            # if cache is available, get_ppl_with_cache
            all_prefix = ["".join(pg[i] for pg in batched_prompts) for i in range(batch_num)]
            prefix_cache, cached_num = self.cacher.get_cache(all_prefix)
            if prefix_cache is not None and cached_num == len(batched_prompts) - 1:
                # logger.warning(f"get_prob_with_cache, {cached_num} {len(batched_prompts)}")
                return self.get_prob_with_cache(batched_prompts[-1], batched_option_nums, prefix_cache)

            # pass the input without prefix text to the model
            concat_cached_prefix = ["".join(pg[i] for pg in batched_prompts[:cached_num + 1]) for i in range(batch_num)]
            prefix_cache = self.get_cache(batched_prompts[cached_num], prefix_cache, save_last_logits=False)

            for p, c in zip(concat_cached_prefix, prefix_cache):
                self.cacher.set_cache(p, c, cached_num)
            return []

        batched_prompts, batched_option_nums = map(list, zip(*batched_inputs))
        batched_encodings = self.tokenizer(
            batched_prompts,
            padding="longest",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            batch_logits = self.model(
                input_ids=batched_encodings["input_ids"].to(self.device),
                attention_mask=batched_encodings["attention_mask"].to(self.device),
            ).logits.detach()[:, -1]  # padding_side="left" in tokenizer

            answers = []
            for i, option_num in enumerate(batched_option_nums):
                label_ids = self._get_label_ids(option_num)
                answers.append(torch.softmax(batch_logits[i, label_ids], dim=-1, dtype=torch.float32).tolist())
        return answers
