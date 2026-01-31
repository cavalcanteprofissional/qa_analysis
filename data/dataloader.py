import pandas as pd
import os
from typing import List, Optional, Dict
from config.settings import Config

class DataLoader:
    """Carregador de dados para os intervalos"""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Inicializa o carregador de dados
        
        Args:
            data_dir: Diretório com os arquivos CSV
        """
        self.data_dir = data_dir if data_dir is not None else Config.INTERVALOS_DIR
        
    def list_intervals(self) -> List[str]:
        """
        Lista todos os intervalos disponíveis
        
        Returns:
            Lista com nomes dos arquivos CSV
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Diretório não encontrado: {self.data_dir}")
        
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        return sorted(csv_files)
    
    def load_interval(self, interval_name: str) -> pd.DataFrame:
        """
        Carrega um intervalo específico
        
        Args:
            interval_name: Nome do arquivo CSV ou número do intervalo
            
        Returns:
            DataFrame com os dados
        """
        # Se for um número, tentar encontrar o arquivo correspondente
        if interval_name.isdigit():
            csv_files = self.list_intervals()
            interval_files = [f for f in csv_files if f.startswith(f"shard_{int(interval_name):03d}")]
            
            if not interval_files:
                raise FileNotFoundError(f"Nenhum arquivo encontrado para intervalo {interval_name}")
            
            interval_name = interval_files[0]
        
        # Garantir extensão .csv
        if not interval_name.endswith('.csv'):
            interval_name += '.csv'
        
        file_path = os.path.join(self.data_dir, interval_name)
        
        if not os.path.exists(file_path):
            available = self.list_intervals()
            raise FileNotFoundError(
                f"Arquivo {interval_name} não encontrado. "
                f"Intervalos disponíveis: {available}"
            )
        
        print(f"📂 Carregando intervalo: {interval_name}")
        
        # Carregar CSV
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Dataset carregado: {len(df)} linhas, {len(df.columns)} colunas")
            print(f"   Colunas: {list(df.columns)}")
            
            # Padronizar nomes de colunas
            if 'text' in df.columns and 'query' in df.columns:
                df = df.rename(columns={'text': 'context', 'query': 'question'})
                print("   Colunas renomeadas: 'text' → 'context', 'query' → 'question'")
            
            return df
            
        except Exception as e:
            raise ValueError(f"Erro ao carregar {file_path}: {e}")
    
    def get_interval_info(self, interval_name: str) -> Dict:
        """
        Retorna informações sobre um intervalo
        
        Args:
            interval_name: Nome do intervalo
            
        Returns:
            Dict com informações
        """
        df = self.load_interval(interval_name)
        
        return {
            "name": interval_name,
            "rows": len(df),
            "columns": list(df.columns),
            "questions_sample": df['question'].head(3).tolist() if 'question' in df.columns else [],
            "context_length_avg": df['context'].apply(lambda x: len(str(x).split())).mean() if 'context' in df.columns else 0,
            "question_length_avg": df['question'].apply(lambda x: len(str(x).split())).mean() if 'question' in df.columns else 0
        }