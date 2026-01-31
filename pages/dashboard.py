"""
Dashboard page - Main overview page
"""
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from utils.data_manager import DataManager

def show_dashboard():
    """Main dashboard page"""
    
    st.markdown('<div class="main-header">📊 QA Analysis Dashboard</div>', unsafe_allow_html=True)
    
    # Initialize data manager
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = DataManager()
    
    # Dashboard overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Dataset Status", "Loaded" if st.session_state.get('data_loaded', False) else "Not Loaded")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Models Ready", "Ready" if st.session_state.get('models_loaded', False) else "Not Ready")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if st.session_state.get('current_data') is not None:
            st.metric("Total Examples", len(st.session_state.current_data))
        else:
            st.metric("Total Examples", "0")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Analysis Status", "Complete" if st.session_state.get('processing_complete', False) else "Pending")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📁 Load Dataset", type="primary", use_container_width=True):
            st.switch_page("app.py?page=Data Management")
    
    with col2:
        if st.button("🤖 Configure Models", use_container_width=True):
            st.switch_page("app.py?page=Model Analysis")
    
    with col3:
        if st.button("📈 View Results", use_container_width=True):
            st.switch_page("app.py?page=Results Visualization")
    
    st.markdown("---")
    
    # Data preview (if loaded)
    if st.session_state.get('data_loaded', False) and st.session_state.get('current_data') is not None:
        st.subheader("📋 Dataset Preview")
        
        df = st.session_state.current_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Dataset Information:**
            - Name: {st.session_state.get('dataset_name', 'Unknown')}
            - Rows: {len(df):,}
            - Columns: {len(df.columns)}
            - Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB
            """)
        
        with col2:
            if st.session_state.get('dataset_info'):
                info = st.session_state.dataset_info
                st.info(f"""
                **Text Statistics:**
                - Avg Context Length: {info.get('context_length_avg', 0):.1f} words
                - Avg Question Length: {info.get('question_length_avg', 0):.1f} words
                """)
        
        # Sample data display
        with st.expander("👁️ View Sample Data", expanded=True):
            display_cols = ['question', 'context'] if 'question' in df.columns else df.columns[:2]
            st.dataframe(
                df[display_cols].head(10),
                use_container_width=True,
                hide_index=True
            )
    
    # Recent exports (if any)
    if st.session_state.get('export_directory'):
        st.subheader("📂 Recent Exports")
        export_path = st.session_state.export_directory
        
        if os.path.exists(export_path):
            export_files = [f for f in os.listdir(export_path) if f.endswith('.csv')]
            
            if export_files:
                st.success(f"Found {len(export_files)} export files")
                
                for file in export_files[:5]:  # Show first 5 files
                    file_path = os.path.join(export_path, file)
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    st.write(f"📄 **{file}** - {file_size:.1f} KB ({file_time.strftime('%Y-%m-%d %H:%M')})")
            else:
                st.info("No export files found")
    
    # Instructions section
    if not st.session_state.get('data_loaded', False):
        st.subheader("📖 Getting Started")
        
        st.markdown("""
        ### Welcome to the QA Analysis Dashboard!
        
        This dashboard provides comprehensive analysis and comparison capabilities for Question Answering models.
        
        **Getting Started:**
        1. **Load Dataset**: Upload your own CSV/RAR files or browse existing datasets
        2. **Configure Models**: Choose between DistilBERT and RoBERTa models  
        3. **Run Analysis**: Process your data and compare model performance
        4. **View Results**: Explore comprehensive visualizations and export results
        
        **Supported Features:**
        - 🤖 **Multiple Models**: DistilBERT (fast) and RoBERTa (accurate)
        - 📊 **Comprehensive Metrics**: Confidence scores, overlap analysis, model comparisons
        - 📈 **Rich Visualizations**: Charts, tables, statistical analysis
        - 💾 **Export Capabilities**: Multiple CSV formats with detailed analysis
        - 🔄 **Hybrid Interface**: Upload files or browse existing datasets
        """)