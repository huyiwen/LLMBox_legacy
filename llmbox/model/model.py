from logging import getLogger
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from tiktoken import Encoding
from transformers import (PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast)
from vllm import LLM

if TYPE_CHECKING:
    # solve the circular import
    from ..utils import ModelArguments
    from ..utils.cache_prefix_sampler import Cacher

logger = getLogger(__name__)


class Model:
    r"""The base model object for all models.

    Args:
        args (ModelArguments): The global configurations.

    Attributes:
        name (str): The name of this model.
        type (str): The type of this model, which can be chosen from `base` and `instruction`.
        tokenizer (Union[transformers.PreTrainedTokenizer, PreTrainedTokenizerFast, tiktoken.Encoding]): The tokenizer of this model.
        max_tokens (int): The maximum token length of this model.
        generation_kwargs (dict): The configurations for open-ended generation.
        ppl_kwargs (dict, *optional*): The configurations for computing PPL score.
    """
    name = ""
    type = ""

    model: Union[PreTrainedModel, LLM, None] = None
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, Encoding]
    cacher: Optional["Cacher"] = None

    def __init__(self, args: "ModelArguments"):
        self.args = args

    def set_cacher(self, cacher: "Cacher"):
        r"""Set the cacher for this model. The cacher is used to cache the generated results for the model."""
        self.cacher = cacher

    def set_ppl_args(self, **extra_model_args):
        r"""Set the configurations for PPL score calculation. This is useful because different datasets may have different requirements for ppl calculation."""
        raise NotImplementedError(f"{self.name} model must implement the `set_ppl_args` function.")

    def get_ppl(self, batched_inputs: List[Tuple[str, ...]]) -> List[Tuple[float, int]]:
        r"""Compute the PPL score of the option given the context for this batch.

        Args:
            batched_inputs (List[Tuple[str, str]]): The batch of context and option pairs.

        Returns:
            List[float]: The list of PPL scores.
        """
        raise NotImplementedError(f"{self.name} model does not support `get_ppl`.")

    def set_generation_args(self, **extra_model_args):
        r"""Set the configurations for open-ended generation. This is useful because different datasets may have different requirements for generation."""

        raise NotImplementedError(f"{self.name} model must implement the `set_generation_args` function.")

    def generation(self, batched_inputs: List[str]) -> List[str]:
        r"""Generate the response of given question for this batch.

        Args:
            batched_inputs (List[str]): The batch of questions.

        Returns:
            List[str]: The list of generation results.
        """
        raise NotImplementedError(f"{self.name} model does not support `generation`.")

    def set_prob_args(self, **extra_model_args):
        r"""Set the configurations for generation. This is useful because different datasets may have different requirements for get_prob."""

        raise NotImplementedError(f"{self.name} model must implement the `set_prob_args` function.")

    def get_prob(self, batched_inputs: List[Tuple[str, int]]) -> List[int]:
        r"""Calculates the probability of each option in a multiple-choice question being the correct continuation of a given text prompt.

        Args:
            batched_inputs (List[Tuple[str, int]]): The batch of questions and corresponding option nums.

        Returns:
            List[int]: The option index of maximal probabiltiy.
        """
        raise NotImplementedError(f"{self.name} model does not support `get_prob`.")
