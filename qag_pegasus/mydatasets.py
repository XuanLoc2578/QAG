from torch.utils.data import Dataset
from transformers.models.pegasus.tokenization_pegasus_fast import PegasusTokenizerFast
from sklearn.model_selection import train_test_split
import pandas as pd
import pytorch_lightning as pl

SEP_QA_TOKEN = "[SEP]"


def read_and_split_data(data_file: str):
    df = pd.read_csv(data_file, skipinitialspace=True)

    df["qa0"] = df["question0"] + f" {SEP_QA_TOKEN} " + df["answer0"]
    df["qa1"] = df["question1"] + f" {SEP_QA_TOKEN} " + df["answer1"]
    df["qa2"] = df["question2"] + f" {SEP_QA_TOKEN} " + df["answer2"]
    df["qa3"] = df["question3"] + f" {SEP_QA_TOKEN} " + df["answer3"]
    df = df.dropna()
    train_df, test_df = train_test_split(df, test_size=0.2)
    test_df, val_df = train_test_split(test_df, test_size=0.5)
    return train_df, test_df, val_df


class SquadDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: PegasusTokenizerFast,
        source_max_token_len: int = 256,
        target_max_token_len: int = 64
    ):

        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]

        source_encoding = self.tokenizer(
            data_row["context"],
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            [data_row["qa0"], data_row["qa1"], data_row["qa2"], data_row["qa3"]],
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        labels = target_encoding["input_ids"]
        labels[labels == 0] = -100.0

        return dict(
            input_ids=source_encoding["input_ids"].flatten(),
            attention_mask=source_encoding["attention_mask"].flatten(),
            labels=labels
        )


class SquadDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: PegasusTokenizerFast,
        batch_size: int = 2,
        source_max_token_len: int = 256,
        target_max_token_len: int = 64
    ):

        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def setup(self, stage=None):
        self.train_dataset = SquadDataset(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )

        self.val_dataset = SquadDataset(
            self.val_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )

        self.test_dataset = SquadDataset(
            self.test_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )
