from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.models.pegasus.tokenization_pegasus_fast import PegasusTokenizerFast
from qag_pegasus.min_ref_loss_model import CustomPegasusForConditionalGeneration
from qag_pegasus.mydatasets import (
    read_and_split_data,
    SEP_QA_TOKEN,
    SquadDataModule
)
from qag_pegasus.training_agruments import ModelArguments, DataTrainingArguments
from transformers.hf_argparser import HfArgumentParser
import logging
import torch

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = PegasusTokenizerFast.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    tokenizer.add_special_tokens({'additional_special_tokens': [SEP_QA_TOKEN]})
    tokenizer_len = len(tokenizer)

    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CustomPegasusForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    ).to(torch_device)

    model.resize_token_embeddings(tokenizer_len)

    if model_args.freeze_encoder:
        for param in model.model.encoder.parameters():
            param.requires_grad = False

    train_df, test_df, val_df = read_and_split_data(data_args.train_file)
    data_module = SquadDataModule(train_df, val_df, test_df, tokenizer,
                                  batch_size=training_args.per_device_train_batch_size)
    data_module.setup()

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=data_module.train_dataset,
        tokenizer=tokenizer
    )
    trainer.train()
    # save model and tokenizer
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    return trainer


if __name__ == "__main__":
    main()
