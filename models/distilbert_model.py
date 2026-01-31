from models.base_model import BaseQAModel
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, pipeline
import torch
from huggingface_hub import login
from config.settings import Config

class DistilBERTModel(BaseQAModel):
    """Implementação do modelo DistilBERT para QA"""
    
    def __init__(self, config: Dict = None):
        """
        Inicializa o modelo DistilBERT
        
        Args:
            config: Configurações específicas do modelo
        """
        default_config = Config.MODEL_CONFIG["distilbert"]
        if config:
            default_config.update(config)
            
        super().__init__(
            model_name=Config.MODELS["distilbert"],
            config=default_config
        )
        
    def load_model(self):
        """Carrega o modelo DistilBERT"""
        print(f"⚙️ Carregando modelo DistilBERT...")
        
        # Login no Hugging Face
        login(token=Config.HF_TOKEN)
        
        # Usar pipeline do transformers
        self.model = pipeline(
            "question-answering",
            model=self.model_name,
            tokenizer=self.model_name,
            device=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        print(f"✅ DistilBERT carregado com sucesso!")
        print(f"   Dispositivo: {self.device}")
        print(f"   Configurações: {self.config}")
        
    def predict(self, question: str, context: str) -> Dict:
        """
        Faz predição para uma única pergunta-contexto
        
        Args:
            question: Texto da pergunta
            context: Texto do contexto
            
        Returns:
            Dict com resposta e score
        """
        if self.model is None:
            self.load_model()
        
        try:
            # Limitar contexto para evitar overflow
            max_length = self.config.get("max_length", 512)
            truncated_context = context[:max_length]
            
            result = self.model({
                "question": question,
                "context": truncated_context
            })
            
            return {
                "answer": result["answer"],
                "score": result["score"],
                "start": result["start"],
                "end": result["end"]
            }
            
        except Exception as e:
            print(f"⚠️ Erro na predição: {e}")
            return {
                "answer": "",
                "score": 0.0,
                "start": 0,
                "end": 0
            }