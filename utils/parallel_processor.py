"""
Parallel Model Processing System for QA Analysis
Handles multi-threaded model processing with comprehensive error logging
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import threading
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any
import json
import os

from transformers import pipeline
from config.settings import Config
from utils.data_manager import ErrorLogger
from utils.helpers import HelperFunctions

class ModelProcessor:
    """Individual model processing with error handling"""
    
    def __init__(self, model_name: str, model_path: str, device: int = -1):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.pipeline = None
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """Load the model pipeline"""
        try:
            self.pipeline = pipeline(
                "question-answering",
                model=self.model_path,
                tokenizer=self.model_path,
                device=self.device
            )
            self.is_loaded = True
            return True
        except Exception as e:
            st.error(f"Failed to load {self.model_name}: {str(e)}")
            return False
    
    def process_example(self, question: str, context: str, example_id: int = None) -> Dict:
        """Process a single example"""
        if not self.is_loaded:
            return {
                'error': 'Model not loaded',
                'answer': '',
                'score': 0.0,
                'start': 0,
                'end': 0
            }
        
        try:
            # Limit context to prevent token overflow
            truncated_context = context[:512]
            
            # Process with model
            result = self.pipeline({
                'question': question,
                'context': truncated_context
            })
            
            # Calculate overlap
            overlap = HelperFunctions.calculate_overlap(context, result['answer'])
            
            return {
                'answer': result['answer'],
                'score': result['score'],
                'start': result['start'],
                'end': result['end'],
                'overlap': overlap,
                'truncated_context': truncated_context != context
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'answer': '',
                'score': 0.0,
                'start': 0,
                'end': 0,
                'overlap': 0.0
            }

class ParallelProcessingManager:
    """Manages parallel processing of multiple models"""
    
    def __init__(self, max_workers: int = 2):
        self.max_workers = max_workers
        self.error_logger = ErrorLogger()
        self.processors = {}
        self.processing_stats = {}
        
    def initialize_models(self, models_to_load: List[str]) -> Dict[str, bool]:
        """Initialize selected models"""
        results = {}
        
        for model_name in models_to_load:
            model_path = Config.MODELS.get(model_name)
            if not model_path:
                results[model_name] = False
                continue
            
            device = 0 if torch.cuda.is_available() else -1
            processor = ModelProcessor(model_name, model_path, device)
            
            success = processor.load_model()
            results[model_name] = success
            
            if success:
                self.processors[model_name] = processor
                self.processing_stats[model_name] = {
                    'total_processed': 0,
                    'errors': 0,
                    'total_score': 0.0,
                    'total_overlap': 0.0
                }
        
        self.error_logger.log_info(f"Models initialized: {list(results.keys())}, Success: {list(results.values())}")
        return results
    
    def process_dataset_parallel(self, df: pd.DataFrame, models: List[str], 
                             progress_callback=None) -> Dict[str, Any]:
        """
        Process dataset with parallel models
        
        Args:
            df: Dataset to process
            models: List of model names to use
            progress_callback: Function to update progress
            
        Returns:
            Processing results and statistics
        """
        
        if not models:
            raise ValueError("No models specified for processing")
        
        start_time = time.time()
        self.error_logger.log_info(f"Starting parallel processing: {len(df)} examples, models: {models}")
        
        # Initialize results storage
        results = {
            'metadata': {
                'start_time': start_time,
                'total_examples': len(df),
                'models_used': models,
                'device': 'CUDA' if torch.cuda.is_available() else 'CPU'
            },
            'results': [],
            'errors': [],
            'statistics': {}
        }
        
        # Process each example
        for idx, row in df.iterrows():
            if progress_callback:
                progress = (idx + 1) / len(df)
                progress_callback(progress, f"Processing example {idx + 1}/{len(df)}")
            
            question = str(row['question'])
            context = str(row['context'])
            
            example_result = {
                'example_id': idx,
                'question': question,
                'context': context,
                'model_results': {}
            }
            
            # Process with each model
            for model_name in models:
                if model_name in self.processors:
                    try:
                        model_result = self.processors[model_name].process_example(
                            question, context, idx
                        )
                        
                        # Update statistics
                        stats = self.processing_stats[model_name]
                        stats['total_processed'] += 1
                        stats['total_score'] += model_result.get('score', 0.0)
                        stats['total_overlap'] += model_result.get('overlap', 0.0)
                        
                        if 'error' in model_result:
                            stats['errors'] += 1
                            results['errors'].append({
                                'example_id': idx,
                                'model': model_name,
                                'error': model_result['error'],
                                'question': question[:100] + '...' if len(question) > 100 else question
                            })
                        
                        example_result['model_results'][model_name] = model_result
                        
                    except Exception as e:
                        self.error_logger.log_error(
                            f"Error processing example {idx} with {model_name}",
                            e,
                            {'question': question[:100]}
                        )
                        
                        example_result['model_results'][model_name] = {
                            'error': str(e),
                            'answer': '',
                            'score': 0.0,
                            'overlap': 0.0
                        }
            
            results['results'].append(example_result)
        
        # Calculate final statistics
        end_time = time.time()
        results['metadata']['end_time'] = end_time
        results['metadata']['processing_time'] = end_time - start_time
        
        for model_name in models:
            if model_name in self.processing_stats:
                stats = self.processing_stats[model_name]
                if stats['total_processed'] > 0:
                    results['statistics'][model_name] = {
                        'total_processed': stats['total_processed'],
                        'errors': stats['errors'],
                        'error_rate': stats['errors'] / stats['total_processed'],
                        'avg_score': stats['total_score'] / stats['total_processed'],
                        'avg_overlap': stats['total_overlap'] / stats['total_processed']
                    }
        
        self.error_logger.log_processing_step("parallel_processing_complete", {
            'total_examples': len(df),
            'processing_time': results['metadata']['processing_time'],
            'total_errors': len(results['errors']),
            'models': models
        })
        
        return results
    
    def convert_to_dataframe(self, processing_results: Dict) -> pd.DataFrame:
        """Convert processing results to DataFrame format"""
        df_list = []
        
        for result in processing_results['results']:
            row = {
                'example_id': result['example_id'],
                'question': result['question'],
                'context': result['context']
            }
            
            # Add model-specific columns
            for model_name, model_result in result['model_results'].items():
                row[f'{model_name}_answer'] = model_result.get('answer', '')
                row[f'{model_name}_score'] = model_result.get('score', 0.0)
                row[f'{model_name}_overlap'] = model_result.get('overlap', 0.0)
                row[f'{model_name}_error'] = model_result.get('error', None)
            
            df_list.append(row)
        
        return pd.DataFrame(df_list)
    
    def export_processing_results(self, results: Dict, export_dir: str) -> str:
        """Export processing results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(export_dir, exist_ok=True)
        
        # Export main results
        df = self.convert_to_dataframe(results)
        main_file = os.path.join(export_dir, f"processing_results_{timestamp}.csv")
        df.to_csv(main_file, index=False)
        
        # Export metadata
        metadata_file = os.path.join(export_dir, f"processing_metadata_{timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(results['metadata'], f, indent=2)
        
        # Export statistics
        stats_file = os.path.join(export_dir, f"processing_statistics_{timestamp}.json")
        with open(stats_file, 'w') as f:
            json.dump(results['statistics'], f, indent=2)
        
        # Export errors if any
        if results['errors']:
            errors_df = pd.DataFrame(results['errors'])
            errors_file = os.path.join(export_dir, f"processing_errors_{timestamp}.csv")
            errors_df.to_csv(errors_file, index=False)
        
        self.error_logger.log_info(f"Processing results exported to: {export_dir}")
        return export_dir

class StreamlitProgressTracker:
    """Progress tracking for Streamlit interface"""
    
    def __init__(self):
        self.progress_bar = None
        self.status_text = None
        self.current_step = 0
        self.total_steps = 1
    
    def setup(self, total_steps: int = 1, description: str = "Processing..."):
        """Setup progress tracking"""
        self.total_steps = total_steps
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.status_text.text(description)
    
    def update(self, current_step: int, message: str = ""):
        """Update progress"""
        self.current_step = current_step
        progress = min(current_step / self.total_steps, 1.0)
        
        if self.progress_bar:
            self.progress_bar.progress(progress)
        
        if self.status_text:
            if message:
                self.status_text.text(message)
            else:
                self.status_text.text(f"Step {current_step}/{self.total_steps}")
    
    def finish(self, message: str = "Complete!"):
        """Mark as complete"""
        self.update(self.total_steps, message)

# Streamlit integration functions
def run_parallel_analysis(df: pd.DataFrame, selected_models: List[str], 
                        progress_tracker: StreamlitProgressTracker = None) -> Dict:
    """
    Run parallel analysis with Streamlit integration
    
    Args:
        df: Dataset to process
        selected_models: List of models to use
        progress_tracker: Progress tracker for UI updates
        
    Returns:
        Processing results
    """
    
    # Initialize processing manager
    processing_manager = ParallelProcessingManager()
    
    # Setup progress tracking
    if progress_tracker:
        progress_tracker.setup(3, "Initializing models...")
    
    # Initialize models
    progress_tracker.update(1, "Loading models...") if progress_tracker else None
    model_results = processing_manager.initialize_models(selected_models)
    
    # Check if models loaded successfully
    failed_models = [model for model, success in model_results.items() if not success]
    if failed_models:
        st.error(f"Failed to load models: {failed_models}")
        return None
    
    successful_models = [model for model, success in model_results.items() if success]
    
    # Process dataset
    progress_tracker.update(2, "Processing dataset...") if progress_tracker else None
    processing_results = processing_manager.process_dataset_parallel(
        df, successful_models, 
        lambda p, m: progress_tracker.update(2 + p/len(df), m) if progress_tracker else None
    )
    
    # Export results
    progress_tracker.update(3, "Exporting results...") if progress_tracker else None
    export_dir = os.path.join(Config.OUTPUT_DIR, "parallel_processing")
    processing_manager.export_processing_results(processing_results, export_dir)
    
    # Mark as complete
    if progress_tracker:
        progress_tracker.finish("Processing complete!")
    
    return processing_results