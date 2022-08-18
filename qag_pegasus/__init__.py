from transformers.models.pegasus.tokenization_pegasus_fast import PegasusTokenizerFast
from qag_pegasus.min_ref_loss_model import CustomPegasusForConditionalGeneration
import unicodedata as ud
import torch


class QAGPegasus:
    def __init__(self, model_name_or_path: str):
        self.tokenizer = PegasusTokenizerFast.from_pretrained(model_name_or_path)
        self.model = CustomPegasusForConditionalGeneration.from_pretrained(model_name_or_path)

    @staticmethod
    def normalize(text):
        text = ud.normalize("NFC", text)
        text = " ".join(text.split())
        return text

    def generate_qa(
        self,
        context: str,
        num_return_sequences=4
    ):
        context = self.normalize(context)
        inputs = self.tokenizer(context, return_tensors="pt")
        outputs = self.model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=64,
            do_sample=True,
            top_k=num_return_sequences,
            num_return_sequences=num_return_sequences,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
        )
        outputs = self.tokenizer.batch_decode(outputs)
        outputs = [s.replace("<pad>", "").strip() for s in outputs]
        return outputs
