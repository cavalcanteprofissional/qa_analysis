from abc import ABC, abstractmethod
from transformers import pipeline
import torch
from typing import Dict, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm

class BaseQAModel(ABC):
    """Classe base abstrata para modelos de QA"""
    
    def __init__(self, model_name: str, config: Dict):
        """
        Inicializa o modelo base
        
        Args:
            model_name: Nome do modelo no Hugging Face
            config: Configurações do modelo
        """
        self.model_name = model_name
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
    @abstractmethod
    def load_model(self):
        """Carrega o modelo e tokenizer"""
        pass
    
    @abstractmethod
    def predict(self, question: str, context: str) -> Dict:
        """
        Faz predição para uma única pergunta-contexto
        
        Args:
            question: Texto da pergunta
            context: Texto do contexto
            
        Returns:
            Dict com resposta e score
        """
        pass
    
    def predict_batch(self, questions: List[str], contexts: List[str]) -> List[Dict]:
        """
        Faz predição para um batch de perguntas-contextos
        
        Args:
            questions: Lista de perguntas
            contexts: Lista de contextos
            
        Returns:
            Lista de dicionários com respostas e scores
        """
        results = []
        for question, context in zip(questions, contexts):
            result = self.predict(question, context)
            results.append(result)
        return results
    
    def evaluate_dataset(self, df: pd.DataFrame, 
                        question_col: str = "question",
                        context_col: str = "context") -> pd.DataFrame:
        """
        Avalia um dataset completo
        
        Args:
            df: DataFrame com perguntas e contextos
            question_col: Nome da coluna de perguntas
            context_col: Nome da coluna de contextos
            
        Returns:
            DataFrame com resultados
        """
        if self.model is None:
            self.load_model()
        
        results = []
        print(f"🔍 Avaliando {len(df)} exemplos com {self.model_name}...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                result = self.predict(
                    question=str(row[question_col]),
                    context=str(row[context_col])[:self.config.get("max_length", 512)]
                )
                
                results.append({
                    "_id": row.get("_id", idx),
                    "question": row[question_col],
                    "context": row[context_col],
                    "answer": result["answer"],
                    "score": result["score"],
                    "start": result.get("start", 0),
                    "end": result.get("end", 0)
                })
                
            except Exception as e:
                print(f"⚠️ Erro no exemplo {idx}: {e}")
                results.append({
                    "_id": row.get("_id", idx),
                    "question": row[question_col],
                    "context": row[context_col],
                    "answer": "",
                    "score": 0.0,
                    "start": 0,
                    "end": 0
                })
        
        return pd.DataFrame(results)
    
    def get_info(self) -> Dict:
        """Retorna informações sobre o modelo"""
        return {
            "name": self.model_name,
            "config": self.config,
            "device": self.device,
            "loaded": self.model is not None
        }