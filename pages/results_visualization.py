"""
Results Visualization page - Display analysis results and export visualizations
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
from utils.import_export import ImportExportManager
from utils.data_manager import DataManager
from config.settings import Config

def show_results_visualization():
    """Results visualization page for displaying analysis results"""
    
    st.markdown('<div class="main-header">📈 Results Visualization</div>', unsafe_allow_html=True)
    
    # Initialize managers
    if 'import_export_manager' not in st.session_state:
        st.session_state.import_export_manager = ImportExportManager()
    
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = DataManager()
    
    # Check if results are available
    if not st.session_state.get('processing_complete', False):
        st.warning("⚠️ No analysis results available. Please run model analysis first.")
        return
    
    # Results tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Summary Dashboard", 
        "📈 Model Comparison", 
        "📋 Detailed Results",
        "📁 Export Browser"
    ])
    
    with tab1:
        show_summary_dashboard()
    
    with tab2:
        show_model_comparison()
    
    with tab3:
        show_detailed_results()
    
    with tab4:
        show_export_browser()

def show_summary_dashboard():
    """Show summary dashboard with key metrics"""
    
    st.subheader("📊 Analysis Summary")
    
    results = st.session_state.analysis_results
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_examples = len(results.get('distilbert_answers', results.get('roberta_answers', [])))
    
    with col1:
        st.metric("Total Examples", total_examples)
    
    with col2:
        if results.get('distilbert_scores'):
            avg_score = np.mean(results['distilbert_scores'])
            st.metric("Avg DistilBERT Score", f"{avg_score:.3f}")
    
    with col3:
        if results.get('roberta_scores'):
            avg_score = np.mean(results['roberta_scores'])
            st.metric("Avg RoBERTa Score", f"{avg_score:.3f}")
    
    with col4:
        processing_time = results.get('processing_time', 'N/A')
        st.metric("Processing Time", processing_time)
    
    st.markdown("---")
    
    # Score distributions
    st.subheader("📈 Score Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if results.get('distilbert_scores'):
            fig_distilbert = px.histogram(
                x=results['distilbert_scores'],
                nbins=20,
                title="DistilBERT Score Distribution",
                labels={'x': 'Confidence Score', 'y': 'Count'}
            )
            fig_distilbert.update_layout(height=300)
            st.plotly_chart(fig_distilbert, use_container_width=True)
    
    with col2:
        if results.get('roberta_scores'):
            fig_roberta = px.histogram(
                x=results['roberta_scores'],
                nbins=20,
                title="RoBERTa Score Distribution",
                labels={'x': 'Confidence Score', 'y': 'Count'}
            )
            fig_roberta.update_layout(height=300)
            st.plotly_chart(fig_roberta, use_container_width=True)
    
    # Overlap analysis
    if results.get('distilbert_overlaps') and results.get('roberta_overlaps'):
        st.subheader("🔗 Overlap Analysis")
        
        overlap_data = pd.DataFrame({
            'DistilBERT': results['distilbert_overlaps'],
            'RoBERTa': results['roberta_overlaps']
        })
        
        fig_overlap = make_subplots(
            rows=1, cols=2,
            subplot_titles=('DistilBERT Overlap Distribution', 'RoBERTa Overlap Distribution')
        )
        
        fig_overlap.add_trace(
            go.Histogram(x=results['distilbert_overlaps'], name='DistilBERT'),
            row=1, col=1
        )
        
        fig_overlap.add_trace(
            go.Histogram(x=results['roberta_overlaps'], name='RoBERTa'),
            row=1, col=2
        )
        
        fig_overlap.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_overlap, use_container_width=True)

def show_model_comparison():
    """Show detailed model comparison"""
    
    st.subheader("📈 Model Comparison Analysis")
    
    results = st.session_state.analysis_results
    
    if not results.get('distilbert_scores') or not results.get('roberta_scores'):
        st.warning("⚠️ Both models need to have results for comparison.")
        return
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Example': range(1, len(results['distilbert_scores']) + 1),
        'DistilBERT_Score': results['distilbert_scores'],
        'RoBERTa_Score': results['roberta_scores'],
        'DistilBERT_Overlap': results['distilbert_overlaps'],
        'RoBERTa_Overlap': results['roberta_overlaps']
    })
    
    # Score comparison scatter plot
    st.subheader("🎯 Score Comparison")
    
    fig_scatter = px.scatter(
        comparison_df,
        x='DistilBERT_Score',
        y='RoBERTa_Score',
        title='DistilBERT vs RoBERTa Scores',
        labels={
            'DistilBERT_Score': 'DistilBERT Confidence Score',
            'RoBERTa_Score': 'RoBERTa Confidence Score'
        },
        hover_data=['Example']
    )
    
    # Add diagonal line
    min_score = min(comparison_df['DistilBERT_Score'].min(), comparison_df['RoBERTa_Score'].min())
    max_score = max(comparison_df['DistilBERT_Score'].max(), comparison_df['RoBERTa_Score'].max())
    
    fig_scatter.add_shape(
        type="line",
        x0=min_score, y0=min_score,
        x1=max_score, y1=max_score,
        line=dict(color="red", dash="dash")
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Overlap comparison
    st.subheader("🔗 Overlap Comparison")
    
    fig_overlap_scatter = px.scatter(
        comparison_df,
        x='DistilBERT_Overlap',
        y='RoBERTa_Overlap',
        title='DistilBERT vs RoBERTa Overlap Scores',
        labels={
            'DistilBERT_Overlap': 'DistilBERT Overlap (%)',
            'RoBERTa_Overlap': 'RoBERTa Overlap (%)'
        },
        hover_data=['Example']
    )
    
    st.plotly_chart(fig_overlap_scatter, use_container_width=True)
    
    # Performance summary table
    st.subheader("📊 Performance Summary")
    
    performance_stats = pd.DataFrame({
        'Metric': ['Mean Score', 'Std Score', 'Mean Overlap', 'Std Overlap', 'Max Score', 'Min Score'],
        'DistilBERT': [
            f"{np.mean(results['distilbert_scores']):.3f}",
            f"{np.std(results['distilbert_scores']):.3f}",
            f"{np.mean(results['distilbert_overlaps']):.3f}",
            f"{np.std(results['distilbert_overlaps']):.3f}",
            f"{np.max(results['distilbert_scores']):.3f}",
            f"{np.min(results['distilbert_scores']):.3f}"
        ],
        'RoBERTa': [
            f"{np.mean(results['roberta_scores']):.3f}",
            f"{np.std(results['roberta_scores']):.3f}",
            f"{np.mean(results['roberta_overlaps']):.3f}",
            f"{np.std(results['roberta_overlaps']):.3f}",
            f"{np.max(results['roberta_scores']):.3f}",
            f"{np.min(results['roberta_scores']):.3f}"
        ]
    })
    
    st.dataframe(performance_stats, use_container_width=True, hide_index=True)
    
    # Winner analysis
    st.subheader("🏆 Model Performance Comparison")
    
    distilbert_wins = sum(1 for d, r in zip(results['distilbert_scores'], results['roberta_scores']) if d > r)
    roberta_wins = sum(1 for d, r in zip(results['distilbert_scores'], results['roberta_scores']) if r > d)
    ties = len(results['distilbert_scores']) - distilbert_wins - roberta_wins
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("DistilBERT Wins", f"{distilbert_wins} ({distilbert_wins/len(results['distilbert_scores'])*100:.1f}%)")
    
    with col2:
        st.metric("RoBERTa Wins", f"{roberta_wins} ({roberta_wins/len(results['roberta_scores'])*100:.1f}%)")
    
    with col3:
        st.metric("Ties", f"{ties} ({ties/len(results['distilbert_scores'])*100:.1f}%)")

def show_detailed_results():
    """Show detailed results with interactive tables"""
    
    st.subheader("📋 Detailed Results")
    
    # Load the complete results dataframe if available
    if st.session_state.get('export_directory'):
        export_dir = st.session_state.export_directory
        csv_files = [f for f in os.listdir(export_dir) if 'resultados_completos' in f and f.endswith('.csv')]
        
        if csv_files:
            results_file = os.path.join(export_dir, csv_files[0])
            results_df = pd.read_csv(results_file)
            
            # Display options
            st.subheader("🔍 Display Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                display_count = st.slider("Show Examples", min_value=5, max_value=100, value=20)
            
            with col2:
                sort_by = st.selectbox(
                    "Sort By",
                    options=['Original Order', 'DistilBERT Score', 'RoBERTa Score', 'Score Difference'],
                    help="Sort examples by selected metric"
                )
            
            with col3:
                filter_threshold = st.slider(
                    "Min Score Filter",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.1,
                    help="Show only examples above this score threshold"
                )
            
            # Apply sorting and filtering
            display_df = results_df.copy()
            
            if sort_by == 'DistilBERT Score' and 'distilbert_score' in display_df.columns:
                display_df = display_df.sort_values('distilbert_score', ascending=False)
            elif sort_by == 'RoBERTa Score' and 'roberta_score' in display_df.columns:
                display_df = display_df.sort_values('roberta_score', ascending=False)
            elif sort_by == 'Score Difference' and 'score_difference' in display_df.columns:
                score_diff_abs = display_df['score_difference'].abs()
                sort_index = score_diff_abs.sort_values(ascending=False).index
                display_df = display_df.reindex(sort_index)
            
            # Apply threshold filter
            score_cols = [col for col in ['distilbert_score', 'roberta_score'] if col in display_df.columns]
            if score_cols and filter_threshold > 0:
                # Convert to numpy for proper boolean indexing
                score_values = display_df[score_cols].values
                max_scores = np.max(score_values, axis=1)
                mask = max_scores >= filter_threshold
                display_df = display_df[mask]
            
            # Display the table
            st.subheader(f"👁️ Showing {min(display_count, len(display_df))} Examples")
            
            # Select columns to display
            display_cols = []
            if 'question' in display_df.columns:
                display_cols.append('question')
            if 'context' in display_df.columns:
                display_cols.append('context')
            if 'distilbert_answer' in display_df.columns:
                display_cols.extend(['distilbert_answer', 'distilbert_score'])
            if 'roberta_answer' in display_df.columns:
                display_cols.extend(['roberta_answer', 'roberta_score'])
            
            if display_cols:
                # Truncate context for better display
                display_df_display = display_df.head(display_count)[display_cols].copy()
                
                if 'context' in display_df_display.columns:
                    display_df_display['context'] = display_df_display['context'].apply(
                        lambda x: str(x)[:200] + "..." if len(str(x)) > 200 else str(x)
                    )
                
                st.dataframe(display_df_display, use_container_width=True, hide_index=True)
            else:
                st.dataframe(display_df.head(display_count), use_container_width=True, hide_index=True)
            
            # Export current view
            st.subheader("💾 Export Current View")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Download Current View as CSV"):
                    csv_data = display_df.head(display_count).to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"qa_results_view_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📊 Generate Summary Report"):
                    generate_summary_report(display_df.head(display_count))
        else:
            st.info("No detailed results file found. Please run analysis first.")
    else:
        st.info("No export directory found. Please run analysis first.")

def show_export_browser():
    """Browse and visualize export files using import/export manager"""
    
    st.subheader("📁 Export File Browser")
    
    import_export_manager = st.session_state.import_export_manager
    
    # Browse analysis files
    with st.expander("📂 Browse Output Directory", expanded=True):
        directory = st.text_input(
            "Directory Path",
            value=Config.OUTPUT_DIR,
            help="Directory containing analysis files"
        )
        
        if st.button("🔍 Browse Directory"):
            with st.spinner("Browsing analysis files..."):
                browse_results = import_export_manager.browse_analysis_files(directory)
                st.session_state.browse_results = browse_results
    
    # Display browse results
    if 'browse_results' in st.session_state:
        browse_results = st.session_state.browse_results
        
        # Summary
        if browse_results['summary']:
            summary = browse_results['summary']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Files", summary['total_files'])
            
            with col2:
                st.metric("Models Detected", len(summary['models_detected']))
            
            with col3:
                st.metric("Datasets", len(summary['datasets_info']))
            
            with col4:
                st.metric("Directory", browse_results['directory'])
        
        # File selection
        if browse_results['files']:
            st.subheader("📋 Available Files")
            
            # Create file selection interface
            file_options = [f['name'] for f in browse_results['files']]
            selected_file = st.selectbox("Select file to load:", file_options)
            
            if selected_file:
                # Find selected file info
                selected_info = next(f for f in browse_results['files'] if f['name'] == selected_file)
                
                # File details
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Size (MB)", f"{selected_info['size_mb']:.2f}")
                
                with col2:
                    st.metric("Type", selected_info['type'])
                
                with col3:
                    st.metric("Modified", selected_info['modified_time'].strftime('%Y-%m-%d %H:%M'))
                
                # Load and visualize file
                if st.button(f"📂 Load {selected_file}"):
                    with st.spinner(f"Loading {selected_file}..."):
                        loaded_data = import_export_manager.load_analysis_package(selected_info['path'])
                        st.session_state.loaded_analysis = loaded_data
                        
                        if 'error' not in loaded_data:
                            st.success(f"✅ Successfully loaded {selected_file}")
                        else:
                            st.error(f"❌ Error loading file: {loaded_data['error']}")
    
    # Display loaded analysis
    if 'loaded_analysis' in st.session_state:
        loaded_data = st.session_state.loaded_analysis
        
        if 'error' not in loaded_data:
            display_loaded_analysis(loaded_data)
        else:
            st.error(f"Failed to load file: {loaded_data['error']}")

def display_loaded_analysis(loaded_data):
    """Display loaded analysis data"""
    
    st.subheader("📊 Loaded Analysis Data")
    
    # File metadata
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("File Type", loaded_data.get('file_type', 'unknown'))
    
    with col2:
        st.metric("Data Type", loaded_data.get('data_type', 'unknown'))
    
    with col3:
        if 'rows' in loaded_data:
            st.metric("Rows", f"{loaded_data['rows']:,}")
    
    # Display data based on type
    data = loaded_data.get('data')
    
    if data is None:
        st.warning("No data to display")
        return
    
    if loaded_data.get('data_type') == 'csv' and isinstance(data, pd.DataFrame):
        st.subheader("📋 Data Preview")
        
        # Display options
        col1, col2 = st.columns(2)
        
        with col1:
            display_rows = st.slider("Display rows", 5, 50, 10)
        
        with col2:
            show_columns = st.multiselect(
                "Select columns to display",
                options=data.columns.tolist(),
                default=data.columns[:5].tolist()
            )
        
        # Display filtered data
        if show_columns:
            st.dataframe(data[show_columns].head(display_rows), use_container_width=True)
        else:
            st.dataframe(data.head(display_rows), use_container_width=True)
        
        # Column info
        with st.expander("📊 Column Information"):
            col_info = pd.DataFrame({
                'Column': data.columns,
                'Data Type': data.dtypes.astype(str),
                'Non-Null Count': data.count(),
                'Unique Values': data.nunique()
            })
            st.dataframe(col_info, use_container_width=True, hide_index=True)
        
        # Download option
        csv_data = data.to_csv(index=False)
        st.download_button(
            label="📥 Download Data",
            data=csv_data,
            file_name=f"{loaded_data['filename']}_extracted.csv",
            mime="text/csv"
        )
    
    elif loaded_data.get('data_type') == 'zip':
        st.subheader("📦 Analysis Package Contents")
        
        if isinstance(data, dict):
            # List contained files
            for filename, content in data.items():
                with st.expander(f"📄 {filename}"):
                    if isinstance(content, pd.DataFrame):
                        st.write(f"DataFrame with {len(content)} rows and {len(content.columns)} columns")
                        st.dataframe(content.head(5), use_container_width=True)
                        
                        # Download option for this file
                        csv_data = content.to_csv(index=False)
                        st.download_button(
                            label=f"📥 Download {filename}",
                            data=csv_data,
                            file_name=filename,
                            mime="text/csv"
                        )
                    
                    elif isinstance(content, dict):
                        st.json(content)
                    
                    else:
                        st.write(f"Content type: {type(content)}")
        
        # Download entire package
        if st.button("📦 Download Full Package"):
            st.info("Use the original ZIP file for the complete package")
    
    elif loaded_data.get('data_type') == 'json':
        st.subheader("📄 JSON Data")
        st.json(data)

def generate_summary_report(df):
    """Generate a summary report from the dataframe"""
    
    st.subheader("📊 Summary Report")
    
    report_lines = []
    report_lines.append("# QA Analysis Summary Report")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append(f"Total Examples: {len(df)}")
    report_lines.append("")
    
    # Add statistics if available
    if 'distilbert_score' in df.columns:
        report_lines.append("## DistilBERT Performance")
        report_lines.append(f"- Average Score: {df['distilbert_score'].mean():.3f}")
        report_lines.append(f"- Max Score: {df['distilbert_score'].max():.3f}")
        report_lines.append(f"- Min Score: {df['distilbert_score'].min():.3f}")
        report_lines.append("")
    
    if 'roberta_score' in df.columns:
        report_lines.append("## RoBERTa Performance")
        report_lines.append(f"- Average Score: {df['roberta_score'].mean():.3f}")
        report_lines.append(f"- Max Score: {df['roberta_score'].max():.3f}")
        report_lines.append(f"- Min Score: {df['roberta_score'].min():.3f}")
        report_lines.append("")
    
    # Sample examples
    report_lines.append("## Sample Examples")
    
    for i, (_, row) in enumerate(df.head(3).iterrows()):
        report_lines.append(f"### Example {i+1}")
        report_lines.append(f"**Question:** {row.get('question', 'N/A')}")
        if 'distilbert_answer' in df.columns:
            report_lines.append(f"**DistilBERT Answer:** {row['distilbert_answer']} (Score: {row['distilbert_score']:.3f})")
        if 'roberta_answer' in df.columns:
            report_lines.append(f"**RoBERTa Answer:** {row['roberta_answer']} (Score: {row['roberta_score']:.3f})")
        report_lines.append("")
    
    report_text = "\n".join(report_lines)
    st.markdown(report_text)
    
    # Download report
    st.download_button(
        label="📄 Download Report as Markdown",
        data=report_text,
        file_name=f"qa_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )