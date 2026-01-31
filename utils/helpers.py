import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os
import re

class HelperFunctions:
    """Funções auxiliares para o sistema QA"""
    
    @staticmethod
    def clean_text(text):
        """Limpa o texto removendo caracteres especiais e normalizando"""
        if not isinstance(text, str):
            return ""
        # Remover caracteres especiais, manter letras, números e espaços
        text = re.sub(r'[^\w\s\.\,\-\?]', ' ', text)
        # Remover múltiplos espaços
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    
    @staticmethod
    def calculate_overlap(context, answer):
        """Calcula a sobreposição de palavras entre contexto e resposta"""
        if not answer or not context:
            return 0

        context_words = set(HelperFunctions.clean_text(context).split())
        answer_words = set(HelperFunctions.clean_text(answer).split())

        if not answer_words:
            return 0

        # Calcular Jaccard similarity
        intersection = len(context_words.intersection(answer_words))
        return intersection / len(answer_words) if len(answer_words) > 0 else 0
    
    @staticmethod
    def setup_plot_style():
        """Configura estilo dos gráficos"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    @staticmethod
    def create_output_dataframes(df_results: pd.DataFrame,
                                model1_name: str,
                                model2_name: str) -> Dict[str, pd.DataFrame]:
        """
        Cria todos os DataFrames de saída para análise
        
        Args:
            df_results: DataFrame com resultados completos
            model1_name: Nome do primeiro modelo
            model2_name: Nome do segundo modelo
            
        Returns:
            Dict com todos os DataFrames
        """
        dataframes = {}
        
        # 1. DataFrame principal
        dataframes['resultados_completos'] = df_results.copy()
        
        # 2. Top/Bottom por modelo
        from utils.metrics import MetricsCalculator
        
        # DistilBERT
        distilbert_top, distilbert_bottom = MetricsCalculator.get_top_bottom_examples(
            df_results, 'distilbert_score'
        )
        distilbert_top['model'] = 'DistilBERT'
        distilbert_bottom['model'] = 'DistilBERT'
        dataframes['distilbert_top'] = distilbert_top
        dataframes['distilbert_bottom'] = distilbert_bottom
        
        # RoBERTa
        roberta_top, roberta_bottom = MetricsCalculator.get_top_bottom_examples(
            df_results, 'roberta_score'
        )
        roberta_top['model'] = 'RoBERTa'
        roberta_bottom['model'] = 'RoBERTa'
        dataframes['roberta_top'] = roberta_top
        dataframes['roberta_bottom'] = roberta_bottom
        
        # 3. Top/Bottom globais
        df_results['best_score'] = df_results[['distilbert_score', 'roberta_score']].max(axis=1)
        df_results['worst_score'] = df_results[['distilbert_score', 'roberta_score']].min(axis=1)
        
        global_top = df_results.nlargest(10, 'best_score').copy()
        global_top['rank'] = range(1, 11)
        global_top['category'] = 'global_top'
        dataframes['global_top'] = global_top
        
        global_bottom = df_results.nsmallest(10, 'worst_score').copy()
        global_bottom['rank'] = range(1, 11)
        global_bottom['category'] = 'global_bottom'
        dataframes['global_bottom'] = global_bottom
        
        # 4. Discordâncias
        model1_cols = {
            'answer': 'distilbert_answer',
            'score': 'distilbert_score'
        }
        
        model2_cols = {
            'answer': 'roberta_answer',
            'score': 'roberta_score'
        }
        
        disagreements = MetricsCalculator.get_disagreement_examples(
            df_results, model1_cols, model2_cols
        )
        
        if len(disagreements) > 0:
            dataframes['disagreements'] = disagreements
        
        # 5. Resumo estatístico
        metrics = MetricsCalculator.calculate_metrics(
            df_results, model1_cols, model2_cols
        )
        
        summary_data = {
            'Metric': list(metrics.keys()),
            'Value': [f"{v:.4f}" if isinstance(v, float) else str(v) for v in metrics.values()]
        }
        
        dataframes['summary'] = pd.DataFrame(summary_data)
        
        return dataframes
    
    @staticmethod
    def export_to_csv(dataframes: Dict[str, pd.DataFrame], 
                     output_dir: str,
                     timestamp: str):
        """
        Exporta todos os DataFrames para CSV
        
        Args:
            dataframes: Dict com DataFrames
            output_dir: Diretório de saída
            timestamp: Timestamp para nome dos arquivos
        """
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = []
        for name, df in dataframes.items():
            if len(df) > 0:  # Só exportar se tiver dados
                filename = f"{name}_{timestamp}.csv"
                filepath = os.path.join(output_dir, filename)
                df.to_csv(filepath, index=False, encoding='utf-8')
                exported_files.append({
                    'name': name,
                    'filename': filename,
                    'rows': len(df),
                    'columns': len(df.columns)
                })
                print(f"✅ {filename}: {len(df)} linhas, {len(df.columns)} colunas")
        
        # Criar índice
        index_df = pd.DataFrame(exported_files)
        index_path = os.path.join(output_dir, f"index_{timestamp}.csv")
        index_df.to_csv(index_path, index=False, encoding='utf-8')
        
        return exported_files
    
    @staticmethod
    def create_visualizations(df_results: pd.DataFrame, output_dir: str, timestamp: str):
        """
        Cria visualizações dos resultados
        
        Args:
            df_results: DataFrame com resultados
            output_dir: Diretório de saída
            timestamp: Timestamp para nome dos arquivos
        """
        HelperFunctions.setup_plot_style()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Distribuição dos scores
        axes[0, 0].hist(df_results['distilbert_score'], bins=30, alpha=0.7, label='DistilBERT')
        axes[0, 0].hist(df_results['roberta_score'], bins=30, alpha=0.7, label='RoBERTa')
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Frequência')
        axes[0, 0].set_title('Distribuição dos Scores')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Boxplot dos scores
        score_data = [df_results['distilbert_score'], df_results['roberta_score']]
        bp = axes[0, 1].boxplot(score_data, labels=['DistilBERT', 'RoBERTa'], patch_artist=True)
        bp['boxes'][0].set_facecolor('skyblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Boxplot dos Scores')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Comparação scores
        axes[1, 0].scatter(df_results['distilbert_score'], df_results['roberta_score'], 
                          alpha=0.5, s=20)
        axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)  # Linha de igualdade
        axes[1, 0].set_xlabel('Score DistilBERT')
        axes[1, 0].set_ylabel('Score RoBERTa')
        axes[1, 0].set_title('Comparação Direta de Scores')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Vitórias por modelo
        distilbert_wins = (df_results['distilbert_score'] > df_results['roberta_score']).sum()
        roberta_wins = (df_results['roberta_score'] > df_results['distilbert_score']).sum()
        ties = (df_results['distilbert_score'] == df_results['roberta_score']).sum()
        
        wins_data = [distilbert_wins, roberta_wins, ties]
        win_labels = ['DistilBERT', 'RoBERTa', 'Empates']
        
        axes[1, 1].bar(win_labels, wins_data, color=['skyblue', 'lightcoral', 'lightgray'])
        axes[1, 1].set_ylabel('Número de Questões')
        axes[1, 1].set_title('Comparação de Desempenho')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for i, v in enumerate(wins_data):
            axes[1, 1].text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Análise Comparativa de Modelos de QA', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Salvar gráfico
        plot_path = os.path.join(output_dir, f"comparison_plot_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Gráficos salvos em: {plot_path}")