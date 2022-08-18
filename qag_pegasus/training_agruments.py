"""Arguments for training."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch."""

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    config_file: Optional[str] = field(
        default=None, metadata={"help": "Path to config file"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    freeze_encoder: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv file)."}
    )
    source_max_token_len: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum total context's sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    target_max_token_len: Optional[int] = field(
        default=64,
        metadata={
            "help": "The maximum total target's sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
