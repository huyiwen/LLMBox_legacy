import time
from logging import getLogger

import openai
import tiktoken

from ..utils import ModelArguments
from .enum import OPENAI_CHAT_MODELS, OPENAI_INSTRUCTION_MODELS
from .model import Model

logger = getLogger(__name__)


class Openai(Model):
    r"""The model for calling OpenAI APIs.

    Please refer to https://platform.openai.com/docs/models.

    We now support GPT-3 (`babbage-002` and `davinci-002`) and GPT-3.5 series models (`gpt-3.5-turbo-instruct`, `gpt-3.5-turbo`, `gpt-3.5-turbo-1106`, and `gpt-3.5-turbo-16k`).
    """

    def __init__(self, args: ModelArguments):
        super().__init__(args)

        if openai.__version__ != "0.28.1":
            logger.warning(f"OpenAI version is {openai.__version__}, not 0.28.1. Please make sure the version is correct.")

        logger.info(f"Trying to load OpenAI model with api_base='{openai.api_base}'")
        self.api_key = openai.api_key  # the actual api key is used in icl

        self.args = args
        self.name = args.model_name_or_path
        self.type = "instruction" if self.name in OPENAI_INSTRUCTION_MODELS else "base"
        self.tokenizer = tiktoken.get_encoding(tiktoken.encoding_name_for_model(self.name))
        self.max_try_times = 5

    def set_ppl_args(self, **extra_model_args):
        r"""Set the configurations for PPL score calculation."""
        # TODO: GPT-3.5 series models don't support echo and logprobs
        self.ppl_kwargs = dict(echo=True, max_tokens=0, logprobs=0)
        self.multi_turn = extra_model_args.pop("multi_turn", False)

    def set_generation_args(self, **extra_model_args):
        r"""Set the configurations for open-ended generation. This is useful because different datasets may have different requirements for generation."""
        generation_kwargs = {}
        for key in [
            "temperature",
            "top_p",
            "max_tokens",
            "best_of",
            "frequency_penalty",
            "presence_penalty",
            "stop",
        ]:
            # ModelArguments > extra_model_args
            value = getattr(self.args, key, None)
            if value is None:
                value = extra_model_args.pop(key, None)

            if key == "max_tokens" and value is None:
                value = 1024
            if value is not None:
                generation_kwargs[key] = value
        if generation_kwargs.get("temperature", 1) == 0:
            generation_kwargs["seed"] = self.args.seed
        self.generation_kwargs = generation_kwargs
        self.multi_turn = extra_model_args.pop("multi_turn", False)

    def get_ppl(self, batched_inputs):
        prompt = [src + tgt for src, tgt in batched_inputs]
        results = self.request(prompt, self.ppl_kwargs)
        ppls = []
        for result, (src, _) in zip(results, batched_inputs):
            tgt_start = max(1, result["logprobs"]["text_offset"].index(len(src)))  # designed for src=''
            tgt_end = len(result["logprobs"]["text_offset"])
            ppl = -sum(result["logprobs"]["token_logprobs"][tgt_start:])
            ppls.append((ppl, tgt_end - tgt_start))
        return ppls

    def generation(self, batched_inputs):
        results = self.request(batched_inputs, self.generation_kwargs, self.multi_turn)
        answers = []
        for result in results:
            if self.name in OPENAI_CHAT_MODELS:
                answer = result[0]["message"]["content"]
            else:
                answer = result["text"]
            answers.append(answer)
        return [tuple(answers)] if self.multi_turn else answers

    def request(self, prompt, openai_kwargs, multi_turn=False):
        r"""Call the OpenAI API.

        Args:
            prompt (List[str]): The list of input prompts.
            openai_kwargs (dict): The additional calling configurations.
            multi_turn (bool): Default is False. Set to True if multi-turns needed.

        Returns:
            List[dict]: The responsed JSON results.
        """
        for _ in range(self.max_try_times):
            try:
                if self.name in OPENAI_CHAT_MODELS:
                    messages = []
                    results = []
                    parts = prompt[0].split("__SEPARATOR__") if multi_turn else prompt
                    for query in parts:
                        if len(query) == 0:
                            continue
                        messages.append({"role": "user", "content": query})
                        response = openai.ChatCompletion.create(model=self.name, messages=messages, **openai_kwargs)
                        message = response["choices"]
                        results.append(message)
                        messages.append({"role": "assistant", "content": message[0]["message"]["content"]})
                    return results
                else:
                    response = openai.Completion.create(model=self.name, prompt=prompt, **openai_kwargs)
                    return response["choices"]
            except openai.error.RateLimitError:
                logger.warning("Receive openai.error.RateLimitError, retrying...")
                time.sleep(10)
            except openai.error.AuthenticationError as e:
                raise e
            except openai.error.InvalidRequestError as e:
                raise e
            except Exception as e:
                logger.warning(f"Receive {e.__class__.__name__}: {str(e)}")
                logger.warning("retrying...")
                time.sleep(1)
        raise ConnectionError("OpenAI API error")
