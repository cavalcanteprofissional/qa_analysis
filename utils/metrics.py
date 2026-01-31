import re
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

class MetricsCalculator:
    """Calculadora de métricas para avaliação de QA"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Limpa o texto removendo caracteres especiais
        
        Args:
            text: Texto a ser limpo
            
        Returns:
            Texto limpo
        """
        if not isinstance(text, str):
            return ""
        
        # Remover caracteres especiais, manter letras, números e espaços
        text = re.sub(r'[^\w\s\.\,\-\?]', ' ', text)
        # Remover múltiplos espaços
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    
    @staticmethod
    def calculate_overlap(context: str, answer: str) -> float:
        """
        Calcula a sobreposição de palavras entre contexto e resposta
        
        Args:
            context: Texto do contexto
            answer: Texto da resposta
            
        Returns:
            Overlap (0 a 1)
        """
        if not answer or not context:
            return 0.0
        
        context_words = set(MetricsCalculator.clean_text(context).split())
        answer_words = set(MetricsCalculator.clean_text(answer).split())
        
        if not answer_words:
            return 0.0
        
        intersection = len(context_words.intersection(answer_words))
        return intersection / len(answer_words)
    
    @staticmethod
    def calculate_metrics(df_results: pd.DataFrame, 
                         model1_cols: Dict,
                         model2_cols: Dict) -> Dict:
        """
        Calcula métricas comparativas entre dois modelos
        
        Args:
            df_results: DataFrame com resultados
            model1_cols: Dict com nomes das colunas do modelo 1
            model2_cols: Dict com nomes das colunas do modelo 2
            
        Returns:
            Dict com métricas calculadas
        """
        metrics = {}
        
        # Scores médios
        metrics['score_mean_model1'] = df_results[model1_cols['score']].mean()
        metrics['score_mean_model2'] = df_results[model2_cols['score']].mean()
        metrics['score_std_model1'] = df_results[model1_cols['score']].std()
        metrics['score_std_model2'] = df_results[model2_cols['score']].std()
        
        # Overlaps médios
        overlaps_model1 = df_results.apply(
            lambda row: MetricsCalculator.calculate_overlap(
                row['context'], 
                row[model1_cols['answer']]
            ), axis=1
        )
        
        overlaps_model2 = df_results.apply(
            lambda row: MetricsCalculator.calculate_overlap(
                row['context'], 
                row[model2_cols['answer']]
            ), axis=1
        )
        
        metrics['overlap_mean_model1'] = overlaps_model1.mean()
        metrics['overlap_mean_model2'] = overlaps_model2.mean()
        
        # Correlações
        metrics['corr_score_overlap_model1'] = df_results[model1_cols['score']].corr(overlaps_model1)
        metrics['corr_score_overlap_model2'] = df_results[model2_cols['score']].corr(overlaps_model2)
        
        # Comparação entre modelos
        metrics['model1_wins'] = (df_results[model1_cols['score']] > df_results[model2_cols['score']]).sum()
        metrics['model2_wins'] = (df_results[model2_cols['score']] > df_results[model1_cols['score']]).sum()
        metrics['ties'] = (df_results[model1_cols['score']] == df_results[model2_cols['score']]).sum()
        
        metrics['win_rate_model1'] = metrics['model1_wins'] / len(df_results)
        metrics['win_rate_model2'] = metrics['model2_wins'] / len(df_results)
        
        # Diferenças
        metrics['score_difference_mean'] = (df_results[model2_cols['score']] - df_results[model1_cols['score']]).mean()
        metrics['score_difference_std'] = (df_results[model2_cols['score']] - df_results[model1_cols['score']]).std()
        
        return metrics
    
    @staticmethod
    def get_top_bottom_examples(df_results: pd.DataFrame, 
                               score_col: str, 
                               n: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retorna top N melhores e piores exemplos
        
        Args:
            df_results: DataFrame com resultados
            score_col: Nome da coluna de score
            n: Número de exemplos
            
        Returns:
            Tuple: (top_df, bottom_df)
        """
        top_df = df_results.nlargest(n, score_col).copy()
        top_df['rank'] = range(1, n + 1)
        top_df['category'] = 'top'
        
        bottom_df = df_results.nsmallest(n, score_col).copy()
        bottom_df['rank'] = range(1, n + 1)
        bottom_df['category'] = 'bottom'
        
        return top_df, bottom_df
    
    @staticmethod
    def get_disagreement_examples(df_results: pd.DataFrame,
                                 model1_cols: Dict,
                                 model2_cols: Dict,
                                 n: int = 5,
                                 exclude_top_bottom: int = 10) -> pd.DataFrame:
        """
        Retorna exemplos onde os modelos discordam significativamente
        
        Args:
            df_results: DataFrame com resultados
            model1_cols: Dict com colunas do modelo 1
            model2_cols: Dict com colunas do modelo 2
            n: Número de exemplos
            exclude_top_bottom: Número de extremos a excluir
            
        Returns:
            DataFrame com discordâncias
        """
        # Identificar índices dos extremos
        top_model1 = df_results.nlargest(exclude_top_bottom, model1_cols['score']).index
        bottom_model1 = df_results.nsmallest(exclude_top_bottom, model1_cols['score']).index
        top_model2 = df_results.nlargest(exclude_top_bottom, model2_cols['score']).index
        bottom_model2 = df_results.nsmallest(exclude_top_bottom, model2_cols['score']).index
        
        extreme_indices = set(top_model1) | set(bottom_model1) | set(top_model2) | set(bottom_model2)
        
        # Calcular se respostas são diferentes
        df_results['answers_differ'] = df_results.apply(
            lambda row: MetricsCalculator.clean_text(row[model1_cols['answer']]) != 
                       MetricsCalculator.clean_text(row[model2_cols['answer']]), 
            axis=1
        )
        
        # Calcular diferença absoluta de score
        df_results['score_diff_abs'] = abs(df_results[model1_cols['score']] - df_results[model2_cols['score']])
        
        # Filtrar exemplos
        disagreement_df = df_results[
            ~df_results.index.isin(extreme_indices) &
            df_results['answers_differ'] &
            (df_results['score_diff_abs'] > 0.1)
        ].copy()
        
        # Ordenar por maior diferença
        disagreement_df = disagreement_df.sort_values('score_diff_abs', ascending=False)
        
        # Selecionar top N
        if len(disagreement_df) > 0:
            result_df = disagreement_df.head(n).copy()
            result_df['rank'] = range(1, len(result_df) + 1)
            result_df['category'] = 'disagreement'
            
            return result_df
        
        return pd.DataFrame()