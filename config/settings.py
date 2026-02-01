import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class Config:
    """Configurações do sistema QA"""
    
    # Diretórios
    DATA_DIR = "./data"
    INTERVALOS_DIR = "./data/intervalos"
    
    # Token do Hugging Face (obter de variável de ambiente)
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    
    # Shard selection from environment with backward compatibility
    _SHARD_SELECTOR = None  # Will be initialized below
    
    @classmethod
    def get_shard_selector(cls):
        """Get current shard selector"""
        if cls._SHARD_SELECTOR is None:
            cls._SHARD_SELECTOR = cls._get_shard_selector()
        return cls._SHARD_SELECTOR
    
    @staticmethod
    def _get_shard_selector():
        """Get shard selector from environment with backward compatibility"""
        shard_selector = os.getenv("shard_selector")
        if shard_selector:
            return shard_selector
        
        # Backward compatibility for old typo
        old_shard_selector = os.getenv("shard_seletor")
        if old_shard_selector:
            return old_shard_selector
        
        # Default fallback
        return "shard_055"
    
    @classmethod
    def get_shard_path(cls, shard_name: str = None) -> Path:
        """Get full path to shard file"""
        shard = shard_name or cls.get_shard_selector()
        if shard and not shard.endswith('.csv'):
            shard = f"{shard}.csv"
        return Path(cls.INTERVALOS_DIR) / shard if shard else Path(cls.INTERVALOS_DIR)
    
    @classmethod
    def validate_shard(cls, shard_name: str = None) -> bool:
        """Validate if shard file exists"""
        shard_path = cls.get_shard_path(shard_name)
        return shard_path.exists()
    
    @classmethod
    def list_available_shards(cls) -> list:
        """List all available shard files"""
        intervalos_dir = Path(cls.INTERVALOS_DIR)
        if not intervalos_dir.exists():
            return []
        
        shards = []
        for file in sorted(intervalos_dir.glob("shard_*.csv")):
            shards.append(file.stem)
        return shards
    
    # Modelos disponíveis
    MODELS = {
        "distilbert": "distilbert/distilbert-base-cased-distilled-squad",
        "roberta": "deepset/roberta-base-squad2"
    }
    
    # Dispositivo de processamento
    DEVICE = "cuda" if os.system("nvidia-smi") == 0 else "cpu"
    
    # Configurações de processamento
    BATCH_SIZE = 8
    MAX_LENGTH = 512
    
    # Configurações de cache
    CACHE_DIR = "./cache"
    ENABLE_CACHE = True
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = "qa_system.log"
    
    # Configurações de exportação
    OUTPUT_DIR = "./output"
    METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
    
    @classmethod
    def update_device_config(cls):
        """Atualiza configuração de dispositivo baseado na disponibilidade"""
        import torch
        cls.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def setup_dirs(cls):
        """Cria diretórios necessários"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.INTERVALOS_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.METRICS_DIR, exist_ok=True)
        cls.update_device_config()