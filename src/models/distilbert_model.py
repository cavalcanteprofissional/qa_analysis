from typing import List, Dict, Any
from ..base_model import BaseQAModel


class DistilBERTModel(BaseQAModel):
    """Wrapper mínimo para DistilBERT QA usando Hugging Face pipeline."""
    def __init__(self, hf_name: str = "distilbert-base-cased-distilled-squad", device: str = "cpu") -> None:
        super().__init__(model_name="distilbert", device=device)
        self.hf_name = hf_name
        self.pipe = None

    def load_model(self) -> None:
        from transformers import pipeline
        import torch

        device_idx = 0 if (self.device == "cuda" and torch.cuda.is_available()) else -1
        self.pipe = pipeline("question-answering", model=self.hf_name, tokenizer=self.hf_name, device=device_idx)

    def predict(self, questions: List[str], contexts: List[str]) -> List[Dict[str, Any]]:
        if self.pipe is None:
            self.load_model()
        inputs = [{"question": q, "context": c} for q, c in zip(questions, contexts)]
        res = self.pipe(inputs)
        if isinstance(res, dict):
            res = [res]
        return res

    def get_metadata(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "hf_name": self.hf_name, "device": self.device}
