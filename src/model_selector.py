"""Seleção e registro de modelos disponíveis."""
from typing import List, Union, Dict, Any


class ModelSelector:
    MODELS_REGISTRY: Dict[str, Dict[str, Any]] = {
        "distilbert": {
            "class": "DistilBERTModel",
            "hf_name": "distilbert-base-cased-distilled-squad",
        },
        "roberta": {
            "class": "RobertaModel",
            "hf_name": "deepset/roberta-base-squad2",
        },
        "bert": {
            "class": "BERTModel",
            "hf_name": "bert-large-uncased-whole-word-masking-finetuned-squad",
        },
    }

    def list_available(self) -> List[str]:
        return list(self.MODELS_REGISTRY.keys())

    def select_models(self, selection: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """Retorna uma lista de descritores de modelos (não instanciados).

        O descritor contém `key` e `hf_name` para uso posterior na criação
        dinâmica do modelo dentro de workers.
        """
        if not selection:
            return []

        if isinstance(selection, str):
            if selection == "all":
                return [
                    {"key": k, **v} for k, v in self.MODELS_REGISTRY.items()
                ]
            selection = [selection]

        selected = []
        for s in selection:
            if s == "all":
                return [{"key": k, **v} for k, v in self.MODELS_REGISTRY.items()]
            if s in self.MODELS_REGISTRY:
                selected.append({"key": s, **self.MODELS_REGISTRY[s]})

        return selected
