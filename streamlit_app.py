#!/usr/bin/env python3
"""
Streamlit Dashboard for QA Analysis Export
Comprehensive interface for question-answering model comparison and analysis
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import project modules
from config.settings import Config

# Page imports
from pages.dashboard import show_dashboard
from pages.data_management import show_data_management
from pages.model_analysis import show_model_analysis
from pages.results_visualization import show_results_visualization

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="QA Analysis Dashboard",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Setup directories
    Config.setup_dirs()
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown('<div class="sidebar-header">🤖 QA Analysis</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        page = st.selectbox(
            "Select Page",
            [
                "📊 Dashboard",
                "📁 Data Management", 
                "🤖 Model Analysis",
                "📈 Results Visualization"
            ]
        )
        
        st.markdown("---")
        
        # Quick info
        st.subheader("Quick Info")
        if st.session_state.get('data_loaded', False):
            st.success(f"✅ Dataset loaded: {st.session_state.get('dataset_name', 'Unknown')}")
        else:
            st.info("ℹ️ No dataset loaded")
            
        if st.session_state.get('models_loaded', False):
            st.success("✅ Models ready")
        else:
            st.warning("⚠️ Models not loaded")
    
    # Page routing
    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "📁 Data Management":
        show_data_management()
    elif page == "🤖 Model Analysis":
        show_model_analysis()
    elif page == "📈 Results Visualization":
        show_results_visualization()

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    
    # Data related state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'dataset_name' not in st.session_state:
        st.session_state.dataset_name = None
    if 'dataset_info' not in st.session_state:
        st.session_state.dataset_info = {}
    
    # Model related state
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    if 'qa_distilbert' not in st.session_state:
        st.session_state.qa_distilbert = None
    if 'qa_roberta' not in st.session_state:
        st.session_state.qa_roberta = None
    
    # Results related state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'export_directory' not in st.session_state:
        st.session_state.export_directory = None

if __name__ == "__main__":
    main()