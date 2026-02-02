"""Base abstracto para modelos de QA.

Define a interface mínima que implementações concretas devem seguir.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import torch


class BaseQAModel(ABC):
    """Interface base para modelos de Question Answering.

    Subclasses devem implementar carregamento, predição em batch e
    metadados.
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device

    @abstractmethod
    def load_model(self) -> None:
        """Carrega pesos / tokenizer e prepara o modelo para inferência."""

    @abstractmethod
    def predict(self, questions: List[str], contexts: List[str]) -> List[Dict[str, Any]]:
        """Executa predições em batch. Retorna lista de dicionários com chaves
        como `answer`, `score`, `start`, `end`.
        """

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Retorna metadados úteis sobre o modelo (nome HF, device, etc.)."""
