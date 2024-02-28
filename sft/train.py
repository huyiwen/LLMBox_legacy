import logging
import warnings
from dataclasses import dataclass
from typing import Optional
from accelerate.utils import set_seed
from sft_dataset import AutoDataset
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
)
from transformers.hf_argparser import HfArg
from typing import Dict, Optional, Sequence
import torch
import transformers

@dataclass
class Arguments(TrainingArguments):
    model_name_or_path: str = HfArg(
        default=None,
        help="The model name or path, e.g., `meta-llama/Llama-2-7b-hf` or `./output/saved_model`",
    )

    tokenizer_name_or_path: Optional[str] = HfArg(
        default=None,
        help="The tokenizer name or path. Default to `model_name_or_path`.",
    )

    data_path: str = HfArg(
        default=None,
        help="The path of dataset, e.g., `data/alpaca_data.json.` or `data/chinese.txt`",
    )

    model_max_length: int = HfArg(
        default=2048,
        help="The maximum sequence length",
    )
    
    IGNORE_INDEX: int = HfArg(
        default=-100,
        help="The index to ignore in the loss function",
    )

    mode: str = HfArg(
        default="sft",
        help="The mode of the training programs, which must be chosen from either `sft` or `pt`.",
        metadata={"choices": ["sft", "pt"]},
    )

    save_only_model: bool = HfArg(
        default=True,
        help="When checkpointing, whether to only save the model, or also the optimizer, scheduler & rng state.",
    )

    use_flash_attention: bool = HfArg(
        default=True,
        help=
        "Whether to use flash attention for a faster and more efficient implementation of the standard attention mechanism.",
    )

    rope_scaling_type: str = HfArg(
        default = "none",
        help="Whether to scaling the RoPE. `none` denotes no scaling of RoPE . `dynamic` and `linear` denoted to scaling RoPE with dynamic NTK and Position Interpolation."
    )

    rope_scaling_factor: int = HfArg(
        default = 4,
        help="Scaling factor of RoPE. The maximum context length will be expanded to the factor times the original maximum positional embedding length."
    )

    bf16: bool = HfArg(
        default=True,
        help="Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
        " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change.",
    )

    tf32: Optional[bool] = HfArg(
        default=True,
        help="Whether to enable tf32 mode, available in Ampere and newer GPU architectures. This is an experimental"
        " API and it may change.",
    )

    lora: Optional[bool] = HfArg(default=False, help="whether to train with LoRA.")

    lora_r: Optional[int] = HfArg(default=16, help='Lora attention dimension (the "rank")')

    lora_alpha: Optional[int] = HfArg(default=16, help="The alpha parameter for Lora scaling.")

    lora_dropout: Optional[float] = HfArg(default=0.05, help="The dropout probability for Lora layers.")
    
    packing: Optional[bool] = HfArg(default=False, help="Whether to pack the input sequences to the maximum length of the model.")


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    args: Arguments
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        if self.args.packing:
            new_input_ids, new_labels = [torch.tensor([], dtype=input_ids[0].dtype)], [torch.tensor([], dtype=input_ids[0].dtype)]
            lengths = [[]]
            for input_id, label in zip(input_ids, labels):
                if len(new_input_ids[-1]) + len(input_id) <= self.tokenizer.model_max_length:
                    new_input_ids[-1] = torch.cat((new_input_ids[-1], input_id))
                    new_labels[-1] = torch.cat((new_labels[-1], label))
                    lengths[-1].append(len(input_id))
                else:
                    new_input_ids.append(input_id)
                    new_labels.append(label)
                    lengths.append([len(input_id)])

            input_ids, labels = new_input_ids, new_labels

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.args.IGNORE_INDEX)

        return dict(
            input_ids=input_ids,
            labels=labels,
        )


def train():
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]

    if args.tokenizer_name_or_path is None:
        args.tokenizer_name_or_path = args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        add_eos_token=True,
        # use_fast=False, # some tokenizer has only one implementation
        legacy=False,  # refer to the issue:https://github.com/huggingface/transformers/pull/24565
        use_cache=False,
    )
    tokenizer.pad_token = tokenizer.unk_token  # for llama-1

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    if args.rope_scaling_type != "none":
        config.rope_scaling = {
            "type" : args.rope_scaling_type,
            "factor" : args.rope_scaling_factor
        }
    if args.use_flash_attention:
        config._attn_implementation = "flash_attention_2"
    else:
        config._attn_implementation = None
    config.use_cache=False
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if args.use_flash_attention else torch.float32,
        config=config
    )
    
    if args.gradient_checkpointing:
        args.gradient_checkpointing_kwargs={'use_reentrant':False} # OR gradient_checkpointing_kwargs={'use_reentrant':True}, please refer to https://github.com/huggingface/transformers/issues/26969

    if args.lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    else:
        peft_config = None

    kwargs = dict(
        model=model,
        args=args,
        tokenizer=tokenizer,
    )
    if args.mode == "sft":
        kwargs.update(
            dict(
                train_dataset=AutoDataset(args, tokenizer),
                data_collator=DataCollatorForSupervisedDataset(args, tokenizer),
            )
        )

    elif args.mode == "pt":
        dataset = load_dataset("text", data_files=args.data_path)["train"]
        model.resize_token_embeddings(len(tokenizer))
        kwargs.update(dict(train_dataset=dataset, packing=True, dataset_text_field="text"))

    trainer = Trainer(**kwargs)
    trainer.train()
    trainer.save_state()


def init():
    set_seed(42)
    warnings.filterwarnings("ignore")
    logging.getLogger("DeepSpeed").setLevel(logging.ERROR)


if __name__ == "__main__":
    init()
    train()
