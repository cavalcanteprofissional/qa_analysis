"""
Import/Export System for QA Analysis Files
Handles reading and writing multi-sheet analysis files with comprehensive metadata
"""

import os
import pandas as pd
import json
import zipfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

from config.settings import Config
from utils.data_manager import ErrorLogger

class AnalysisFileReader:
    """Reader for multi-sheet QA analysis export files"""
    
    def __init__(self):
        self.error_logger = ErrorLogger()
        self.supported_file_types = ['.csv', '.json', '.zip']
    
    def detect_analysis_files(self, directory: str) -> List[Dict]:
        """
        Detect and classify analysis files in directory
        
        Returns:
            List of file information with classification
        """
        files_info = []
        
        if not os.path.exists(directory):
            return files_info
        
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            
            if os.path.isfile(file_path):
                file_info = {
                    'name': file,
                    'path': file_path,
                    'size_mb': os.path.getsize(file_path) / 1024 / 1024,
                    'modified_time': datetime.fromtimestamp(os.path.getmtime(file_path)),
                    'type': self._classify_file(file, file_path)
                }
                
                files_info.append(file_info)
        
        return sorted(files_info, key=lambda x: x['modified_time'], reverse=True)
    
    def _classify_file(self, filename: str, filepath: str) -> str:
        """Classify file type based on name and content"""
        name_lower = filename.lower()
        
        # Main results files
        if any(keyword in name_lower for keyword in ['resultados_completos', 'results', 'main']):
            return 'main_results'
        
        # Top performers
        elif any(keyword in name_lower for keyword in ['top', 'melhores', 'best']):
            return 'top_performers'
        
        # Bottom performers
        elif any(keyword in name_lower for keyword in ['bottom', 'piores', 'worst']):
            return 'bottom_performers'
        
        # Statistics
        elif any(keyword in name_lower for keyword in ['summary', 'resumo', 'statistics', 'stats']):
            return 'statistics'
        
        # Disagreements
        elif any(keyword in name_lower for keyword in ['discordancias', 'disagreement', 'discord']):
            return 'disagreements'
        
        # Index/metadata
        elif any(keyword in name_lower for keyword in ['index', 'metadata', 'info']):
            return 'metadata'
        
        # Unknown type
        else:
            return 'unknown'
    
    def load_analysis_file(self, filepath: str) -> Dict[str, Any]:
        """
        Load analysis file with metadata
        
        Args:
            filepath: Path to analysis file
            
        Returns:
            Dictionary with data and metadata
        """
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            
            file_info = {
                'filepath': filepath,
                'filename': os.path.basename(filepath),
                'size_mb': os.path.getsize(filepath) / 1024 / 1024,
                'loaded_at': datetime.now().isoformat()
            }
            
            # Load based on file extension
            if filepath.endswith('.csv'):
                return self._load_csv_file(filepath, file_info)
            elif filepath.endswith('.json'):
                return self._load_json_file(filepath, file_info)
            elif filepath.endswith('.zip'):
                return self._load_zip_file(filepath, file_info)
            else:
                raise ValueError(f"Unsupported file type: {filepath}")
        
        except Exception as e:
            self.error_logger.log_error(f"Failed to load analysis file: {filepath}", e)
            return {'error': str(e), 'filepath': filepath}
    
    def _load_csv_file(self, filepath: str, file_info: Dict) -> Dict:
        """Load CSV analysis file"""
        df = pd.read_csv(filepath)
        
        return {
            **file_info,
            'data': df,
            'data_type': 'csv',
            'rows': len(df),
            'columns': list(df.columns),
            'file_type': self._classify_file(os.path.basename(filepath), filepath)
        }
    
    def _load_json_file(self, filepath: str, file_info: Dict) -> Dict:
        """Load JSON analysis file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return {
            **file_info,
            'data': data,
            'data_type': 'json',
            'file_type': self._classify_file(os.path.basename(filepath), filepath)
        }
    
    def _load_zip_file(self, filepath: str, file_info: Dict) -> Dict:
        """Load ZIP archive with multiple analysis files"""
        extracted_files = {}
        
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            
            for file in file_list:
                if file.endswith('.csv') or file.endswith('.json'):
                    with zip_ref.open(file) as extracted_file:
                        if file.endswith('.csv'):
                            extracted_files[file] = pd.read_csv(extracted_file)
                        else:
                            content = extracted_file.read().decode('utf-8')
                            extracted_files[file] = json.loads(content)
        
        return {
            **file_info,
            'data': extracted_files,
            'data_type': 'zip',
            'contained_files': file_list,
            'file_type': 'analysis_package'
        }
    
    def create_analysis_summary(self, loaded_files: List[Dict]) -> Dict:
        """Create summary from multiple loaded analysis files"""
        summary = {
            'total_files': len(loaded_files),
            'files_by_type': {},
            'datasets_info': {},
            'models_detected': set(),
            'created_at': datetime.now().isoformat()
        }
        
        for file_data in loaded_files:
            if 'error' in file_data:
                continue
            
            file_type = file_data.get('file_type', 'unknown')
            
            # Count by type
            if file_type not in summary['files_by_type']:
                summary['files_by_type'][file_type] = 0
            summary['files_by_type'][file_type] += 1
            
            # Extract dataset info
            if file_data.get('data_type') == 'csv':
                df = file_data.get('data')
                if df is not None and len(df) > 0:
                    dataset_name = file_data.get('filename', 'unknown')
                    
                    # Detect models from column names
                    columns = df.columns.tolist()
                    if 'distilbert_answer' in columns or 'distilbert_score' in columns:
                        summary['models_detected'].add('DistilBERT')
                    if 'roberta_answer' in columns or 'roberta_score' in columns:
                        summary['models_detected'].add('RoBERTa')
                    
                    summary['datasets_info'][dataset_name] = {
                        'rows': len(df),
                        'columns': len(df),
                        'column_names': columns,
                        'file_type': file_type
                    }
        
        summary['models_detected'] = list(summary['models_detected'])
        return summary

class AnalysisFileWriter:
    """Writer for multi-sheet QA analysis export files"""
    
    def __init__(self):
        self.error_logger = ErrorLogger()
    
    def create_analysis_package(self, results_df: pd.DataFrame, export_dir: str, 
                             analysis_name: str = None) -> str:
        """
        Create comprehensive analysis package with multiple sheets
        
        Args:
            results_df: Main results dataframe
            export_dir: Directory to save files
            analysis_name: Name for the analysis
            
        Returns:
            Path to created package
        """
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_name = analysis_name or f"qa_analysis_{timestamp}"
        
        os.makedirs(export_dir, exist_ok=True)
        
        # Create all analysis sheets
        sheets = self._create_analysis_sheets(results_df)
        
        # Save each sheet
        exported_files = []
        for sheet_name, df_data in sheets.items():
            filename = f"{sheet_name}_{timestamp}.csv"
            filepath = os.path.join(export_dir, filename)
            
            if isinstance(df_data, pd.DataFrame):
                df_data.to_csv(filepath, index=False, encoding='utf-8')
            elif isinstance(df_data, dict):
                with open(filepath, 'w') as f:
                    json.dump(df_data, f, indent=2)
            
            exported_files.append({
                'sheet_name': sheet_name,
                'filename': filename,
                'filepath': filepath,
                'rows': len(df_data) if isinstance(df_data, pd.DataFrame) else 'N/A'
            })
        
        # Create metadata file
        metadata = {
            'analysis_name': analysis_name,
            'timestamp': timestamp,
            'created_at': datetime.now().isoformat(),
            'total_examples': len(results_df),
            'exported_files': exported_files,
            'models_detected': self._detect_models_in_dataframe(results_df)
        }
        
        metadata_file = os.path.join(export_dir, f"metadata_{timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create ZIP package
        zip_filename = f"{analysis_name}_package_{timestamp}.zip"
        zip_filepath = os.path.join(export_dir, zip_filename)
        
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_info in exported_files:
                zipf.write(file_info['filepath'], file_info['filename'])
            
            zipf.write(metadata_file, f"metadata_{timestamp}.json")
        
        self.error_logger.log_info(f"Analysis package created: {zip_filepath}")
        return zip_filepath
    
    def _create_analysis_sheets(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create all analysis sheets from main results dataframe"""
        sheets = {}
        
        # Main results
        sheets['resultados_completos'] = df.copy()
        
        # Top 10 best by score
        if 'distilbert_score' in df.columns:
            distilbert_top = df.nlargest(10, 'distilbert_score').copy()
            distilbert_top['rank'] = range(1, 11)
            distilbert_top['category'] = 'distilbert_top'
            sheets['distilbert_top10_melhores'] = distilbert_top
        
        if 'roberta_score' in df.columns:
            roberta_top = df.nlargest(10, 'roberta_score').copy()
            roberta_top['rank'] = range(1, 11)
            roberta_top['category'] = 'roberta_top'
            sheets['roberta_top10_melhores'] = roberta_top
        
        # Top 10 worst by score
        if 'distilbert_score' in df.columns:
            distilbert_bottom = df.nsmallest(10, 'distilbert_score').copy()
            distilbert_bottom['rank'] = range(1, 11)
            distilbert_bottom['category'] = 'distilbert_bottom'
            sheets['distilbert_top10_piores'] = distilbert_bottom
        
        if 'roberta_score' in df.columns:
            roberta_bottom = df.nsmallest(10, 'roberta_score').copy()
            roberta_bottom['rank'] = range(1, 11)
            roberta_bottom['category'] = 'roberta_bottom'
            sheets['roberta_top10_piores'] = roberta_bottom
        
        # Global top and bottom
        if 'distilbert_score' in df.columns and 'roberta_score' in df.columns:
            df['best_score'] = df[['distilbert_score', 'roberta_score']].max(axis=1)
            df['worst_score'] = df[['distilbert_score', 'roberta_score']].min(axis=1)
            
            global_top = df.nlargest(10, 'best_score').copy()
            global_top['rank'] = range(1, 11)
            global_top['category'] = 'global_top'
            sheets['global_top10_melhores'] = global_top
            
            global_bottom = df.nsmallest(10, 'worst_score').copy()
            global_bottom['rank'] = range(1, 11)
            global_bottom['category'] = 'global_bottom'
            sheets['global_top10_piores'] = global_bottom
        
        # Disagreements
        if 'distilbert_score' in df.columns and 'roberta_score' in df.columns:
            score_diff = abs(df['distilbert_score'] - df['roberta_score'])
            disagreement_threshold = 0.2
            
            disagreements = df[score_diff > disagreement_threshold].copy()
            disagreements['score_difference'] = df['distilbert_score'] - df['roberta_score']
            disagreements['score_difference_abs'] = score_diff
            disagreements['rank'] = range(1, len(disagreements) + 1)
            
            if len(disagreements) > 0:
                sheets['discordancias'] = disagreements
        
        # Statistics summary
        stats = self._calculate_statistics(df)
        stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
        sheets['resumo_estatistico'] = stats_df
        
        return sheets
    
    def _detect_models_in_dataframe(self, df: pd.DataFrame) -> List[str]:
        """Detect which models are present in the dataframe"""
        models = []
        columns = df.columns.tolist()
        
        if 'distilbert_answer' in columns or 'distilbert_score' in columns:
            models.append('DistilBERT')
        if 'roberta_answer' in columns or 'roberta_score' in columns:
            models.append('RoBERTa')
        
        return models
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive statistics from results dataframe"""
        stats = {
            'total_examples': len(df),
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # DistilBERT statistics
        if 'distilbert_score' in df.columns:
            distilbert_scores = df['distilbert_score'].dropna()
            stats.update({
                'distilbert_avg_score': distilbert_scores.mean(),
                'distilbert_max_score': distilbert_scores.max(),
                'distilbert_min_score': distilbert_scores.min(),
                'distilbert_std_score': distilbert_scores.std()
            })
        
        # RoBERTa statistics
        if 'roberta_score' in df.columns:
            roberta_scores = df['roberta_score'].dropna()
            stats.update({
                'roberta_avg_score': roberta_scores.mean(),
                'roberta_max_score': roberta_scores.max(),
                'roberta_min_score': roberta_scores.min(),
                'roberta_std_score': roberta_scores.std()
            })
        
        # Comparison statistics
        if 'distilbert_score' in df.columns and 'roberta_score' in df.columns:
            distilbert_wins = (df['distilbert_score'] > df['roberta_score']).sum()
            roberta_wins = (df['roberta_score'] > df['distilbert_score']).sum()
            ties = (df['distilbert_score'] == df['roberta_score']).sum()
            
            stats.update({
                'distilbert_wins': distilbert_wins,
                'roberta_wins': roberta_wins,
                'ties': ties,
                'distilbert_win_rate': distilbert_wins / len(df),
                'roberta_win_rate': roberta_wins / len(df)
            })
        
        return stats

class ImportExportManager:
    """Main manager for import/export operations"""
    
    def __init__(self):
        self.reader = AnalysisFileReader()
        self.writer = AnalysisFileWriter()
        self.error_logger = ErrorLogger()
    
    def browse_analysis_files(self, directory: str = None) -> Dict:
        """Browse and categorize analysis files"""
        if directory is None:
            directory = Config.OUTPUT_DIR
        
        files_info = self.reader.detect_analysis_files(directory)
        summary = self.reader.create_analysis_summary(files_info)
        
        return {
            'directory': directory,
            'files': files_info,
            'summary': summary
        }
    
    def load_analysis_package(self, filepath: str) -> Dict:
        """Load complete analysis package from file"""
        return self.reader.load_analysis_file(filepath)
    
    def export_current_analysis(self, results_df: pd.DataFrame, 
                             export_name: str = None) -> str:
        """Export current analysis results"""
        export_dir = os.path.join(Config.OUTPUT_DIR, "exports")
        return self.writer.create_analysis_package(results_df, export_dir, export_name)