"""
Data Management System for QA Analysis
Handles caching, validation, and data operations with disk-based storage
"""

import os
import pandas as pd
import pickle
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

from config.settings import Config
from data.dataloader import DataLoader
from utils.helpers import HelperFunctions

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCache:
    """Disk-based caching system for datasets and results"""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.join(Config.OUTPUT_DIR, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _get_cache_path(self, key: str, extension: str = "pkl") -> str:
        """Generate cache file path from key"""
        # Create hash for long keys
        if len(key) > 100:
            key_hash = hashlib.md5(key.encode()).hexdigest()
            key = f"{key[:50]}_{key_hash}"
        
        return os.path.join(self.cache_dir, f"{key}.{extension}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached data"""
        cache_path = self._get_cache_path(key)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"Cache hit: {key}")
                return data
            except Exception as e:
                logger.error(f"Cache read error: {e}")
        
        return None
    
    def set(self, key: str, data: Any, metadata: Dict = None) -> None:
        """Cache data with metadata"""
        cache_path = self._get_cache_path(key)
        meta_path = self._get_cache_path(key, "json")
        
        try:
            # Cache main data
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Cache metadata
            cache_info = {
                'key': key,
                'cached_at': datetime.now().isoformat(),
                'size_bytes': os.path.getsize(cache_path),
                'data_type': type(data).__name__,
                **(metadata or {})
            }
            
            with open(meta_path, 'w') as f:
                json.dump(cache_info, f, indent=2)
            
            logger.info(f"Cache set: {key}")
            
        except Exception as e:
            logger.error(f"Cache write error: {e}")
    
    def list_cached_items(self) -> List[Dict]:
        """List all cached items with metadata"""
        cached_items = []
        
        for file in os.listdir(self.cache_dir):
            if file.endswith('.json'):
                meta_path = os.path.join(self.cache_dir, file)
                try:
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                        cached_items.append(metadata)
                except Exception as e:
                    logger.error(f"Error reading cache metadata {file}: {e}")
        
        return sorted(cached_items, key=lambda x: x.get('cached_at', ''), reverse=True)
    
    def clear(self, pattern: str = None) -> None:
        """Clear cache items matching pattern"""
        cleared = 0
        
        for file in os.listdir(self.cache_dir):
            if pattern is None or pattern in file:
                try:
                    os.remove(os.path.join(self.cache_dir, file))
                    cleared += 1
                except Exception as e:
                    logger.error(f"Error deleting cache file {file}: {e}")
        
        logger.info(f"Cleared {cleared} cache items")

class DataValidator:
    """Data validation and quality checking"""
    
    @staticmethod
    def validate_qa_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate dataset has required QA format"""
        errors = []
        
        # Check required columns
        required_columns = ['question', 'context']
        for col in required_columns:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        # Check data types
        if 'question' in df.columns:
            if not df['question'].dtype == 'object':
                errors.append("Question column should be text/string")
        
        if 'context' in df.columns:
            if not df['context'].dtype == 'object':
                errors.append("Context column should be text/string")
        
        # Check for empty values
        if 'question' in df.columns:
            empty_questions = df['question'].isnull().sum()
            if empty_questions > 0:
                errors.append(f"{empty_questions} empty questions found")
        
        if 'context' in df.columns:
            empty_contexts = df['context'].isnull().sum()
            if empty_contexts > 0:
                errors.append(f"{empty_contexts} empty contexts found")
        
        # Check content quality
        if 'question' in df.columns:
            short_questions = df[df['question'].str.len() < 5].shape[0]
            if short_questions > len(df) * 0.1:  # More than 10% very short questions
                errors.append(f"High number of short questions ({short_questions})")
        
        if 'context' in df.columns:
            short_contexts = df[df['context'].str.len() < 10].shape[0]
            if short_contexts > len(df) * 0.1:  # More than 10% very short contexts
                errors.append(f"High number of short contexts ({short_contexts})")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def get_data_quality_report(df: pd.DataFrame) -> Dict:
        """Generate comprehensive data quality report"""
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'columns': {}
        }
        
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'null_count': df[col].isnull().sum(),
                'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_count': df[col].nunique(),
                'unique_percentage': (df[col].nunique() / len(df)) * 100
            }
            
            # Text-specific metrics
            if df[col].dtype == 'object':
                text_lengths = df[col].str.len()
                col_info.update({
                    'avg_length': text_lengths.mean(),
                    'min_length': text_lengths.min(),
                    'max_length': text_lengths.max(),
                    'empty_strings': (df[col] == '').sum()
                })
            
            report['columns'][col] = col_info
        
        return report

class ErrorLogger:
    """Comprehensive error logging for data pipeline"""
    
    def __init__(self, log_dir: str = None):
        self.log_dir = log_dir or os.path.join(Config.OUTPUT_DIR, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup file logger
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"qa_pipeline_{timestamp}.log")
        
        self.logger = logging.getLogger(f"qa_pipeline_{timestamp}")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.log_file = log_file
    
    def log_info(self, message: str, extra_data: Dict = None):
        """Log info message with optional extra data"""
        self.logger.info(message)
        if extra_data:
            self.logger.debug(f"Extra data: {json.dumps(extra_data, indent=2)}")
    
    def log_error(self, message: str, exception: Exception = None, context: Dict = None):
        """Log error with exception details and context"""
        error_info = {
            'message': message,
            'exception_type': type(exception).__name__ if exception else None,
            'exception_message': str(exception) if exception else None,
            'context': context or {}
        }
        
        self.logger.error(f"ERROR: {message}")
        self.logger.error(f"Exception: {error_info['exception_type']}: {error_info['exception_message']}")
        if context:
            self.logger.error(f"Context: {json.dumps(context, indent=2)}")
    
    def log_processing_step(self, step_name: str, details: Dict):
        """Log processing step with details"""
        self.logger.info(f"STEP: {step_name}")
        self.logger.info(f"Details: {json.dumps(details, indent=2)}")

class DataManager:
    """Unified data management system with caching and error handling"""
    
    def __init__(self):
        self.cache = DataCache()
        self.validator = DataValidator()
        self.error_logger = ErrorLogger()
        self.dataloader = DataLoader()
        
        self.error_logger.log_info("DataManager initialized")
    
    def load_dataset(self, source: str, source_type: str = 'file') -> pd.DataFrame:
        """
        Load dataset with caching and validation
        
        Args:
            source: File path or dataset identifier
            source_type: 'file', 'upload', 'interval'
        
        Returns:
            Validated pandas DataFrame
        """
        cache_key = f"dataset_{source_type}_{source}"
        
        # Try cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            self.error_logger.log_info(f"Loaded dataset from cache: {source}")
            return cached_data
        
        self.error_logger.log_info(f"Loading dataset: {source_type}:{source}")
        
        try:
            # Load data based on source type
            if source_type == 'file':
                df = pd.read_csv(source)
            elif source_type == 'upload':
                df = pd.read_csv(source)
            elif source_type == 'interval':
                df = self.dataloader.load_interval(source)
            else:
                raise ValueError(f"Unknown source type: {source_type}")
            
            # Validate dataset
            is_valid, errors = self.validator.validate_qa_dataset(df)
            
            if not is_valid:
                error_msg = f"Dataset validation failed: {', '.join(errors)}"
                self.error_logger.log_error(error_msg, context={'source': source, 'errors': errors})
                raise ValueError(error_msg)
            
            # Generate quality report
            quality_report = self.validator.get_data_quality_report(df)
            self.error_logger.log_info(f"Dataset quality: {quality_report['total_rows']} rows, {quality_report['memory_usage_mb']:.2f} MB")
            
            # Cache the dataset
            self.cache.set(cache_key, df, {
                'source': source,
                'source_type': source_type,
                'quality_report': quality_report
            })
            
            return df
            
        except Exception as e:
            self.error_logger.log_error(f"Failed to load dataset: {source}", e, {'source_type': source_type})
            raise
    
    def get_cache_info(self) -> Dict:
        """Get information about cached datasets"""
        cached_items = self.cache.list_cached_items()
        
        return {
            'total_cached_items': len(cached_items),
            'cache_directory': self.cache.cache_dir,
            'cached_items': cached_items
        }
    
    def clear_cache(self, pattern: str = None):
        """Clear cached data"""
        self.cache.clear(pattern)
        self.error_logger.log_info(f"Cache cleared with pattern: {pattern}")
    
    def export_dataset_info(self, df: pd.DataFrame, dataset_name: str) -> str:
        """Export dataset information to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        info = {
            'dataset_name': dataset_name,
            'export_time': timestamp,
            'quality_report': self.validator.get_data_quality_report(df),
            'sample_data': df.head(5).to_dict('records') if len(df) > 0 else []
        }
        
        export_file = os.path.join(Config.OUTPUT_DIR, f"dataset_info_{dataset_name}_{timestamp}.json")
        
        with open(export_file, 'w') as f:
            json.dump(info, f, indent=2, default=str)
        
        self.error_logger.log_info(f"Dataset info exported: {export_file}")
        return export_file