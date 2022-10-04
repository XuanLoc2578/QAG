import transformers
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
        num_return_sequences=4,
        max_length=None,
        num_beams=None,
        do_sample=True,
        top_k=None,
        top_p=0.9,
        temperature=0.7,
        no_repeat_ngram_size=2,
        early_stopping=True
    ):
        context = self.normalize(context)
        inputs = self.tokenizer(context, return_tensors="pt")
        outputs = self.model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        outputs = self.tokenizer.batch_decode(outputs)
        outputs = [s.replace("<pad>", "").strip() for s in outputs]
        return outputs
