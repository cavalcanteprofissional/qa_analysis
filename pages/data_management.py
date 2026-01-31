"""
Data Management page - Upload and browse datasets
"""
import streamlit as st
import pandas as pd
import os
import io
import zipfile
from pathlib import Path

# Import project modules
from data.dataloader import DataLoader
from config.settings import Config
from utils.data_manager import DataManager, DataValidator
from utils.import_export import ImportExportManager

def show_data_management():
    """Data management page for uploading and browsing datasets"""
    
    st.markdown('<div class="main-header">📁 Data Management</div>', unsafe_allow_html=True)
    
    # Initialize data manager
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = DataManager()
    
    # Initialize import/export manager
    if 'import_export_manager' not in st.session_state:
        st.session_state.import_export_manager = ImportExportManager()
    
    data_manager = st.session_state.data_manager
    
    # Tabs for different data sources
    tab1, tab2, tab3 = st.tabs(["📤 Upload Data", "📂 Browse Datasets", "👁️ Data Preview"])
    
    with tab1:
        show_upload_interface()
    
    with tab2:
        show_browse_interface()
    
    with tab3:
        show_data_preview()

def show_upload_interface():
    """Interface for uploading data files"""
    
    st.subheader("📤 Upload Dataset")
    st.markdown("Upload CSV or RAR files containing question-context pairs for QA analysis.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'rar'],
        help="Upload CSV files directly or RAR archives containing CSV files"
    )
    
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Process CSV file
        if uploaded_file.name.endswith('.csv'):
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                
                st.subheader("📋 File Information")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Rows", f"{len(df):,}")
                
                with col2:
                    st.metric("Columns", len(df.columns))
                
                with col3:
                    st.metric("Size", f"{uploaded_file.size / 1024 / 1024:.2f} MB")
                
                st.subheader("📋 Column Information")
                st.dataframe(
                    pd.DataFrame({
                        'Column': df.columns,
                        'Data Type': df.dtypes.astype(str),
                        'Non-Null Count': df.count(),
                        'Sample Values': [str(df[col].dropna().head(3).tolist()) for col in df.columns]
                    }),
                    use_container_width=True
                )
                
                # Column mapping
                st.subheader("🔄 Column Mapping")
                st.info("Map your columns to the required format (question, context)")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    question_col = st.selectbox(
                        "Select Question Column",
                        options=df.columns,
                        help="Column containing the questions"
                    )
                
                with col2:
                    context_col = st.selectbox(
                        "Select Context Column", 
                        options=df.columns,
                        help="Column containing the context text"
                    )
                
                # Preview mapping
                st.subheader("👁️ Mapping Preview")
                preview_df = df[[question_col, context_col]].head(5)
                preview_df.columns = ['Question', 'Context']
                st.dataframe(preview_df, use_container_width=True)
                
                # Load data button
                if st.button("🚀 Load This Dataset", type="primary", use_container_width=True):
                    # Rename columns
                    df_mapped = df.rename(columns={
                        question_col: 'question',
                        context_col: 'context'
                    })
                    
                    # Store in session state
                    st.session_state.current_data = df_mapped
                    st.session_state.data_loaded = True
                    st.session_state.dataset_name = uploaded_file.name
                    
                    # Calculate dataset info
                    st.session_state.dataset_info = {
                        'context_length_avg': df_mapped['context'].apply(lambda x: len(str(x).split())).mean(),
                        'question_length_avg': df_mapped['question'].apply(lambda x: len(str(x).split())).mean()
                    }
                    
                    st.success(f"✅ Dataset '{uploaded_file.name}' loaded successfully!")
                    st.rerun()
                
            except Exception as e:
                st.error(f"Error processing CSV file: {str(e)}")
        
        # Handle RAR files (simplified - for now, just show info)
        elif uploaded_file.name.endswith('.rar'):
            st.warning("""
            RAR file upload detected. 
            
            **Note**: RAR extraction in browser requires additional setup. 
            For now, please extract the RAR file locally and upload the CSV file directly.
            
            **Alternative**: Use the "Browse Datasets" tab to access pre-extracted datasets.
            """)

def show_browse_interface():
    """Interface for browsing existing datasets"""
    
    st.subheader("📂 Browse Available Datasets")
    
    try:
        # Initialize dataloader
        dataloader = DataLoader()
        
        # List available intervals
        intervals = dataloader.list_intervals()
        
        if intervals:
            st.success(f"Found {len(intervals)} datasets")
            
            # Dataset selection
            selected_interval = st.selectbox(
                "Select a dataset to load:",
                options=intervals,
                help="Choose a dataset shard to analyze"
            )
            
            if selected_interval:
                try:
                    # Get dataset info
                    info = dataloader.get_interval_info(selected_interval)
                    
                    # Display information
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Rows", f"{info['rows']:,}")
                    
                    with col2:
                        st.metric("Columns", len(info['columns']))
                    
                    with col3:
                        st.metric("Context Length", f"{info['context_length_avg']:.1f} words")
                    
                    # Sample questions
                    if info['questions_sample']:
                        st.subheader("📝 Sample Questions")
                        for i, question in enumerate(info['questions_sample'][:3], 1):
                            st.write(f"{i}. {question}")
                    
                    # Load button
                    if st.button("🚀 Load This Dataset", type="primary", use_container_width=True):
                        # Load full dataset
                        df = dataloader.load_interval(selected_interval)
                        
                        # Store in session state
                        st.session_state.current_data = df
                        st.session_state.data_loaded = True
                        st.session_state.dataset_name = selected_interval
                        st.session_state.dataset_info = info
                        
                        st.success(f"✅ Dataset '{selected_interval}' loaded successfully!")
                        st.rerun()
                
                except Exception as e:
                    st.error(f"Error loading dataset: {str(e)}")
        
        else:
            st.warning("No datasets found in the data directory.")
            st.info("Please upload data files or check the data directory configuration.")
    
    except Exception as e:
        st.error(f"Error accessing datasets: {str(e)}")

def show_data_preview():
    """Preview currently loaded dataset"""
    
    st.subheader("👁️ Data Preview")
    
    if st.session_state.get('data_loaded', False) and st.session_state.get('current_data') is not None:
        df = st.session_state.current_data
        
        # Dataset summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rows", f"{len(df):,}")
        
        with col2:
            st.metric("Columns", len(df.columns))
        
        with col3:
            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        with col4:
            st.metric("Dataset", st.session_state.get('dataset_name', 'Unknown'))
        
        st.markdown("---")
        
        # Column information
        st.subheader("📋 Column Information")
        
        col_info = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        
        st.dataframe(col_info, use_container_width=True)
        
        # Data preview with filtering
        st.subheader("👁️ Sample Data")
        
        # Filtering options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sample_size = st.slider("Sample Size", min_value=5, max_value=100, value=20)
        
        with col2:
            start_row = st.number_input("Start Row", min_value=0, max_value=len(df)-sample_size, value=0)
        
        with col3:
            show_context = st.checkbox("Show Full Context", value=False)
        
        # Display sample
        display_df = df.iloc[start_row:start_row + sample_size].copy()
        
        if 'context' in display_df.columns and not show_context:
            display_df['context'] = display_df['context'].apply(lambda x: str(x)[:200] + "..." if len(str(x)) > 200 else str(x))
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Export options
        st.subheader("💾 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Sample as CSV", use_container_width=True):
                csv = df.head(sample_size).to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"sample_{st.session_state.get('dataset_name', 'data')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("🗑️ Clear Current Data", use_container_width=True):
                st.session_state.data_loaded = False
                st.session_state.current_data = None
                st.session_state.dataset_name = None
                st.session_state.dataset_info = {}
                st.success("Data cleared from session")
                st.rerun()
    
    else:
        st.info("No data currently loaded. Please upload a dataset or browse existing datasets.")