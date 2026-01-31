"""
Model Analysis page - Configure and run QA models
"""
import streamlit as st
import pandas as pd
import torch
import os
from datetime import datetime
from transformers import pipeline
from tqdm import tqdm

# Import project modules
from config.settings import Config
from utils.helpers import HelperFunctions
from utils.data_manager import DataManager
from utils.parallel_processor import ParallelProcessingManager, StreamlitProgressTracker, run_parallel_analysis

def show_model_analysis():
    """Model analysis page for configuring and running QA models"""
    
    st.markdown('<div class="main-header">🤖 Model Analysis</div>', unsafe_allow_html=True)
    
    # Check if data is loaded
    if not st.session_state.get('data_loaded', False):
        st.warning("⚠️ No dataset loaded. Please load a dataset first in the Data Management page.")
        return
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["⚙️ Model Configuration", "🚀 Run Analysis", "📊 Processing Status"])
    
    with tab1:
        show_model_configuration()
    
    with tab2:
        show_run_analysis()
    
    with tab3:
        show_processing_status()

def show_model_configuration():
    """Model configuration interface"""
    
    st.subheader("⚙️ Model Configuration")
    
    # Device information
    device_info = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    st.info(f"🖥️ **Device Available:** {device_info}")
    
    # Model selection
    st.subheader("🤖 Select Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_distilbert = st.checkbox(
            "🏃 DistilBERT (Fast)",
            value=True,
            help="Lightweight, fast model (~250MB). Good for quick analysis."
        )
    
    with col2:
        use_roberta = st.checkbox(
            "🧠 RoBERTa (Accurate)", 
            value=True,
            help="More robust model (~500MB). Better accuracy on complex questions."
        )
    
    if not use_distilbert and not use_roberta:
        st.error("❌ Please select at least one model to continue.")
        return
    
    # Processing parameters
    st.subheader("⚡ Processing Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_examples = st.number_input(
            "Max Examples",
            min_value=1,
            max_value=len(st.session_state.current_data),
            value=min(100, len(st.session_state.current_data)),
            help="Maximum number of examples to process"
        )
    
    with col2:
        batch_size = st.selectbox(
            "Batch Size",
            options=[1, 2, 4, 8, 16],
            index=1,
            help="Number of examples processed at once"
        )
    
    with col3:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Minimum confidence score for answers"
        )
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            max_length = st.number_input(
                "Max Context Length",
                min_value=128,
                max_value=512,
                value=512,
                help="Maximum length of context in tokens"
            )
        
        with col2:
            overlap_threshold = st.slider(
                "Overlap Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Threshold for overlap analysis"
            )
    
    # Model info display
    st.subheader("📋 Model Information")
    
    model_info_data = []
    
    if use_distilbert:
        model_info_data.append({
            'Model': 'DistilBERT',
            'Size': '~250MB',
            'Speed': 'Fast',
            'Accuracy': 'Good',
            'Use Case': 'Quick analysis, limited resources'
        })
    
    if use_roberta:
        model_info_data.append({
            'Model': 'RoBERTa', 
            'Size': '~500MB',
            'Speed': 'Medium',
            'Accuracy': 'Excellent',
            'Use Case': 'Complex questions, high accuracy needed'
        })
    
    if model_info_data:
        st.dataframe(
            pd.DataFrame(model_info_data),
            use_container_width=True,
            hide_index=True
        )
    
    # Load models button
    if st.button("🚀 Load Models", type="primary", use_container_width=True):
        load_models(use_distilbert, use_roberta)

def show_run_analysis():
    """Run analysis interface"""
    
    st.subheader("🚀 Run Analysis")
    
    # Check if models are loaded
    if not st.session_state.get('models_loaded', False):
        st.warning("⚠️ Models not loaded yet. Please configure and load models first.")
        return
    
    # Show loaded models
    loaded_models = []
    if st.session_state.get('qa_distilbert'):
        loaded_models.append("DistilBERT")
    if st.session_state.get('qa_roberta'):
        loaded_models.append("RoBERTa")
    
    st.success(f"✅ Loaded Models: {', '.join(loaded_models)}")
    
    # Dataset info
    df = st.session_state.current_data
    st.info(f"""
    **Dataset Ready:**
    - Total Examples: {len(df):,}
    - Dataset: {st.session_state.get('dataset_name', 'Unknown')}
    """)
    
    # Processing options
    st.subheader("⚙️ Processing Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        process_count = st.number_input(
            "Examples to Process",
            min_value=1,
            max_value=len(df),
            value=min(50, len(df)),
            help="Number of examples to process in this batch"
        )
    
    with col2:
        save_intermediate = st.checkbox(
            "Save Intermediate Results",
            value=True,
            help="Save progress periodically during processing"
        )
    
    # Run analysis button
    if st.button("🎯 Start Analysis", type="primary", use_container_width=True):
        run_analysis(process_count, save_intermediate)

def show_processing_status():
    """Processing status and results"""
    
    st.subheader("📊 Processing Status")
    
    # Check if processing is complete
    if st.session_state.get('processing_complete', False):
        st.success("✅ Analysis Complete!")
        
        if st.session_state.get('analysis_results'):
            results = st.session_state.analysis_results
            
            # Results summary
            st.subheader("📈 Results Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Examples Processed", len(results))
            
            with col2:
                if results.get('distilbert_scores'):
                    avg_score = sum(results['distilbert_scores']) / len(results['distilbert_scores'])
                    st.metric("Avg DistilBERT Score", f"{avg_score:.3f}")
            
            with col3:
                if results.get('roberta_scores'):
                    avg_score = sum(results['roberta_scores']) / len(results['roberta_scores'])
                    st.metric("Avg RoBERTa Score", f"{avg_score:.3f}")
            
            with col4:
                st.metric("Processing Time", results.get('processing_time', 'N/A'))
            
            # Export directory
            if st.session_state.get('export_directory'):
                st.success(f"📂 Results saved to: {st.session_state.export_directory}")
            
            # View results button
            if st.button("📈 View Detailed Results", use_container_width=True):
                st.switch_page("app.py?page=Results Visualization")
    
    else:
        st.info("ℹ️ No analysis completed yet. Please run analysis to see results here.")

def load_models(use_distilbert, use_roberta):
    """Load selected models using parallel processor"""
    
    st.info("🔄 Loading models... This may take a few moments...")
    
    try:
        # Determine models to load
        models_to_load = []
        if use_distilbert:
            models_to_load.append('distilbert')
        if use_roberta:
            models_to_load.append('roberta')
        
        # Initialize parallel processing manager
        processing_manager = ParallelProcessingManager()
        
        # Load models
        with st.spinner("Loading models..."):
            model_results = processing_manager.initialize_models(models_to_load)
        
        # Check results
        failed_models = [model for model, success in model_results.items() if not success]
        
        if failed_models:
            st.error(f"❌ Failed to load models: {failed_models}")
        else:
            st.success("✅ Models loaded successfully!")
            
            # Store models in session state
            if 'distilbert' in model_results and model_results['distilbert']:
                st.session_state.qa_distilbert = processing_manager.processors['distilbert'].pipeline
            
            if 'roberta' in model_results and model_results['roberta']:
                st.session_state.qa_roberta = processing_manager.processors['roberta'].pipeline
            
            st.session_state.models_loaded = True
            st.session_state.processing_manager = processing_manager
            st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.session_state.models_loaded = False

def run_analysis(process_count, save_intermediate):
    """Run actual analysis using parallel processing"""
    
    # Prepare dataset
    df = st.session_state.current_data.head(process_count)
    
    # Determine available models
    available_models = []
    if st.session_state.get('qa_distilbert'):
        available_models.append('distilbert')
    if st.session_state.get('qa_roberta'):
        available_models.append('roberta')
    
    if not available_models:
        st.error("❌ No models available for processing")
        return
    
    # Setup progress tracker
    progress_tracker = StreamlitProgressTracker()
    progress_tracker.setup(total_steps=3, description="Initializing parallel processing...")
    
    try:
        # Run parallel analysis
        st.session_state.processing_manager = ParallelProcessingManager()
        
        # Initialize models
        progress_tracker.update(1, "Loading models...")
        model_results = st.session_state.processing_manager.initialize_models(available_models)
        
        # Process dataset
        progress_tracker.update(2, "Processing dataset with parallel models...")
        processing_results = st.session_state.processing_manager.process_dataset_parallel(
            df, available_models,
            lambda p, m: progress_tracker.update(2 + p/len(df), f"Processing example {int(p*len(df))}/{len(df)}")
        )
        
        # Convert to DataFrame for compatibility
        results_df = st.session_state.processing_manager.convert_to_dataframe(processing_results)
        
        # Convert to old format for compatibility
        legacy_results = {
            'distilbert_answers': results_df['distilbert_answer'].tolist() if 'distilbert_answer' in results_df.columns else [],
            'distilbert_scores': results_df['distilbert_score'].tolist() if 'distilbert_score' in results_df.columns else [],
            'distilbert_overlaps': results_df['distilbert_overlap'].tolist() if 'distilbert_overlap' in results_df.columns else [],
            'roberta_answers': results_df['roberta_answer'].tolist() if 'roberta_answer' in results_df.columns else [],
            'roberta_scores': results_df['roberta_score'].tolist() if 'roberta_score' in results_df.columns else [],
            'roberta_overlaps': results_df['roberta_overlap'].tolist() if 'roberta_overlap' in results_df.columns else [],
            'processing_time': str(processing_results['metadata']['processing_time']).split('.')[0]
        }
        
        # Store results in session state
        st.session_state.analysis_results = legacy_results
        st.session_state.processing_complete = True
        st.session_state.processing_results = processing_results
        st.session_state.results_dataframe = results_df
        
        # Export final results
        export_dir = os.path.join(Config.OUTPUT_DIR, "parallel_processing")
        export_path = st.session_state.processing_manager.export_processing_results(processing_results, export_dir)
        st.session_state.export_directory = export_path
        
        progress_tracker.finish("✅ Parallel analysis complete!")
        
        st.success(f"🎉 Successfully processed {len(df)} examples with {len(available_models)} models in parallel!")
        st.success(f"📂 Results exported to: {export_path}")
        st.balloons()
        
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error during parallel analysis: {str(e)}")
        progress_tracker.setup(1, "Error occurred...")

def save_intermediate_results(df_partial, results_partial):
    """Save intermediate results (placeholder)"""
    # This would save partial results for recovery
    pass

def export_results(df, results):
    """Export results to CSV files"""
    
    # Create export directory
    from datetime import datetime
    import os
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = os.path.join(Config.OUTPUT_DIR, f"qa_analysis_export_{timestamp}")
    os.makedirs(export_dir, exist_ok=True)
    
    # Create results dataframe
    results_df = df.copy()
    
    if results.get('distilbert_answers'):
        results_df['distilbert_answer'] = results['distilbert_answers']
        results_df['distilbert_score'] = results['distilbert_scores']
        results_df['overlap_distilbert'] = results['distilbert_overlaps']
    
    if results.get('roberta_answers'):
        results_df['roberta_answer'] = results['roberta_answers']
        results_df['roberta_score'] = results['roberta_scores']
        results_df['overlap_roberta'] = results['roberta_overlaps']
    
    # Save main results
    main_file = os.path.join(export_dir, f"resultados_completos_{timestamp}.csv")
    results_df.to_csv(main_file, index=False)
    
    # Store export directory
    st.session_state.export_directory = export_dir
    
    return export_dir