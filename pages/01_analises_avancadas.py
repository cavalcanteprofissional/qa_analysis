"""
Página de Análises Avançadas
Visualizações estatísticas avançadas dos dados do pipeline QA
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import from main app
from app import (
    load_csv, find_latest_results_csv, map_columns,
    PLOTLY_AVAILABLE, MATPLOTLIB_AVAILABLE
)

# Try to import visualization libraries
try:
    import plotly.express as px
    PLOTLY_AVAILABLE_PAGE = True
except ImportError:
    PLOTLY_AVAILABLE_PAGE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE_PAGE = True
except ImportError:
    MATPLOTLIB_AVAILABLE_PAGE = False

try:
    from src.color_manager import ColorManager
    _COLOR_MANAGER_AVAILABLE = True
except ImportError:
    _COLOR_MANAGER_AVAILABLE = False


def create_violin_plot(data, value_col, category_col, color_manager=None):
    """Create violin plot showing distribution by category."""
    if PLOTLY_AVAILABLE_PAGE:
        fig = px.violin(data, y=value_col, x=category_col, 
                       box=True, points="all",
                       title=f"Distribuição de {value_col} por {category_col}")
        fig.update_layout(height=500, showlegend=False)
        return fig
    elif MATPLOTLIB_AVAILABLE_PAGE:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.violinplot(data=data, y=value_col, x=category_col, ax=ax)
        ax.set_title(f"Distribuição de {value_col} por {category_col}")
        return fig
    return None


def create_correlation_heatmap(corr_matrix):
    """Create correlation heatmap."""
    if PLOTLY_AVAILABLE_PAGE:
        fig = px.imshow(corr_matrix,
                       text_auto='.2f',
                       aspect="auto",
                       color_continuous_scale='RdBu_r',
                       title="Matriz de Correlação entre Métricas",
                       zmin=-1, zmax=1)
        fig.update_layout(height=600)
        return fig
    elif MATPLOTLIB_AVAILABLE_PAGE:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                   vmin=-1, vmax=1, ax=ax, fmt='.2f')
        ax.set_title("Matriz de Correlação entre Métricas")
        return fig
    return None


def create_advanced_scatter(data, x_col, y_col, color_col=None, title=""):
    """Create scatter plot with trend line."""
    if PLOTLY_AVAILABLE_PAGE:
        fig = px.scatter(data, x=x_col, y=y_col, 
                        color=color_col if color_col in data.columns else None,
                        title=title,
                        trendline="ols",
                        opacity=0.6)
        
        corr = data[x_col].corr(data[y_col])
        fig.add_annotation(
            x=0.05, y=0.95,
            xref="paper", yref="paper",
            text=f"Correlação: {corr:.3f}",
            showarrow=False,
            bgcolor="white",
            bordercolor="black"
        )
        fig.update_layout(height=500)
        return fig
    elif MATPLOTLIB_AVAILABLE_PAGE:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if color_col and color_col in data.columns:
            for cat in data[color_col].unique():
                subset = data[data[color_col] == cat]
                ax.scatter(subset[x_col], subset[y_col], label=cat, alpha=0.6, s=50)
            ax.legend()
        else:
            ax.scatter(data[x_col], data[y_col], alpha=0.6, s=50)
        
        corr = data[x_col].corr(data[y_col])
        ax.set_title(f"{title} - Correlação: {corr:.3f}")
        return fig
    return None


def create_histogram(data, x_col, title, nbins=30):
    """Create histogram with statistics."""
    if PLOTLY_AVAILABLE_PAGE:
        fig = px.histogram(data, x=x_col, nbins=nbins, 
                          title=title,
                          marginal="box",
                          opacity=0.7)
        
        mean_val = data[x_col].mean()
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                     annotation_text=f"Média: {mean_val:.3f}")
        fig.update_layout(height=500)
        return fig
    elif MATPLOTLIB_AVAILABLE_PAGE:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(data[x_col], bins=nbins, alpha=0.7, edgecolor='black')
        
        mean_val = data[x_col].mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2)
        ax.set_title(title)
        ax.legend([f"Média: {mean_val:.3f}"])
        return fig
    return None


# Page configuration
st.set_page_config(page_title="Análises Avançadas - QA Dashboard", layout="wide")
st.title("📈 Análises Avançadas")
st.markdown("---")

# Load data
csv_path = find_latest_results_csv()
if not csv_path:
    st.error("Nenhum arquivo de resultados encontrado em outputs/.")
    st.stop()

try:
    with st.spinner("Carregando dados..."):
        df = load_csv(csv_path)
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.stop()

if df.empty:
    st.warning("Dados vazios.")
    st.stop()

# Initialize ColorManager
color_manager = None
if _COLOR_MANAGER_AVAILABLE:
    color_manager = ColorManager()
    color_manager.load_from_session_state()

# Map columns
cols = list(df.columns)
colmap = map_columns(cols)

# Get column names
scol = colmap.get("score") or ("score" if "score" in cols else None)
ocol = colmap.get("overlap") or ("overlap" if "overlap" in cols else None)
mcol = colmap.get("model") or ("model" if "model" in cols else None)

# Sidebar filters
st.sidebar.header("🔍 Filtros de Análise")

# Score range filter
if scol in df.columns:
    score_min, score_max = st.sidebar.slider(
        "Range de Score",
        min_value=float(df[scol].min()),
        max_value=float(df[scol].max()),
        value=(float(df[scol].min()), float(df[scol].max())),
        step=0.01
    )
else:
    score_min, score_max = 0.0, 1.0

# Question length filter
if "question_length" in df.columns:
    q_len_min, q_len_max = st.sidebar.slider(
        "Range de Tamanho da Pergunta",
        min_value=int(df["question_length"].min()),
        max_value=int(df["question_length"].max()),
        value=(int(df["question_length"].min()), int(df["question_length"].max()))
    )
else:
    q_len_min, q_len_max = 0, 100

# Model filter
if mcol in df.columns:
    models = df[mcol].unique()
    selected_models = st.sidebar.multiselect("Modelos", options=models, default=models)
else:
    selected_models = []

# Apply filters
filt = df.copy()
if scol in df.columns:
    filt = filt[(filt[scol] >= score_min) & (filt[scol] <= score_max)]
if "question_length" in df.columns:
    filt = filt[(filt["question_length"] >= q_len_min) & (filt["question_length"] <= q_len_max)]
if mcol in df.columns and selected_models:
    filt = filt[filt[mcol].isin(selected_models)]

st.info(f"Dados filtrados: {len(filt)} de {len(df)} registros ({len(filt)/len(df)*100:.1f}%)")

# Row 1: Violin Plot + Statistics
st.subheader("📊 Distribuição Estatística por Modelo")
col1, col2 = st.columns([3, 2])

with col1:
    if mcol in filt.columns and scol in filt.columns:
        st.write("**Distribuição de Scores (Violin Plot)**")
        fig_violin = create_violin_plot(filt, scol, mcol, color_manager)
        if fig_violin is not None:
            if PLOTLY_AVAILABLE_PAGE:
                st.plotly_chart(fig_violin, use_container_width=True)
            elif MATPLOTLIB_AVAILABLE_PAGE:
                st.pyplot(fig_violin)
                plt.close(fig_violin)

with col2:
    st.write("**Estatísticas Descritivas**")
    if mcol in filt.columns and scol in filt.columns:
        stats_df = filt.groupby(mcol)[scol].describe().round(4)
        st.dataframe(stats_df, height=400)

# Row 2: Scatter + Histogram
st.markdown("---")
st.subheader("📈 Relações e Distribuições")
col3, col4 = st.columns(2)

with col3:
    if "question_length" in filt.columns and scol in filt.columns:
        st.write("**Tamanho da Pergunta vs Score**")
        fig_scatter = create_advanced_scatter(
            filt, "question_length", scol, 
            color_col=mcol if mcol in filt.columns else None,
            title="Impacto do Tamanho da Pergunta no Score"
        )
        if fig_scatter is not None:
            if PLOTLY_AVAILABLE_PAGE:
                st.plotly_chart(fig_scatter, use_container_width=True)
            elif MATPLOTLIB_AVAILABLE_PAGE:
                st.pyplot(fig_scatter)
                plt.close(fig_scatter)

with col4:
    if ocol in filt.columns:
        st.write("**Distribuição de Overlap**")
        fig_overlap = create_histogram(filt, ocol, "Distribuição de Overlap", nbins=30)
        if fig_overlap is not None:
            if PLOTLY_AVAILABLE_PAGE:
                st.plotly_chart(fig_overlap, use_container_width=True)
            elif MATPLOTLIB_AVAILABLE_PAGE:
                st.pyplot(fig_overlap)
                plt.close(fig_overlap)

# Row 2.5: Score vs Overlap
st.markdown("---")
st.subheader("🎯 Score vs Overlap")

if scol in filt.columns and ocol in filt.columns:
    col_s_o_1, col_s_o_2 = st.columns([3, 1])
    
    with col_s_o_1:
        st.write("**Score vs Overlap**")
        fig_score_overlap = create_advanced_scatter(
            filt, ocol, scol,
            color_col=mcol if mcol in filt.columns else None,
            title="Relação entre Score e Overlap"
        )
        if fig_score_overlap is not None:
            if PLOTLY_AVAILABLE_PAGE:
                st.plotly_chart(fig_score_overlap, use_container_width=True)
            elif MATPLOTLIB_AVAILABLE_PAGE:
                st.pyplot(fig_score_overlap)
                plt.close(fig_score_overlap)
    
    with col_s_o_2:
        st.write("**Estatísticas**")
        if ocol in filt.columns and scol in filt.columns:
            corr_val = filt[ocol].corr(filt[scol])
            st.metric("Correlação Score-Overlap", f"{corr_val:.4f}")
            
            avg_overlap_high_score = filt[filt[scol] >= 0.8][ocol].mean() if len(filt[filt[scol] >= 0.8]) > 0 else 0
            avg_overlap_low_score = filt[filt[scol] < 0.5][ocol].mean() if len(filt[filt[scol] < 0.5]) > 0 else 0
            
            st.metric("Overlap médio (score ≥ 0.8)", f"{avg_overlap_high_score:.4f}")
            st.metric("Overlap médio (score < 0.5)", f"{avg_overlap_low_score:.4f}")
else:
    st.warning("Colunas de Score ou Overlap não disponíveis.")

# Row 3: Correlation Heatmap
st.markdown("---")
st.subheader("🔗 Matriz de Correlação")

numeric_cols = []
if scol in filt.columns:
    numeric_cols.append(scol)
if ocol in filt.columns:
    numeric_cols.append(ocol)
if "question_length" in filt.columns:
    numeric_cols.append("question_length")
if "context_length" in filt.columns:
    numeric_cols.append("context_length")

if len(numeric_cols) >= 2:
    corr_matrix = filt[numeric_cols].corr()
    
    col5, col6 = st.columns([3, 2])
    
    with col5:
        fig_heatmap = create_correlation_heatmap(corr_matrix)
        if fig_heatmap is not None:
            if PLOTLY_AVAILABLE_PAGE:
                st.plotly_chart(fig_heatmap, use_container_width=True)
            elif MATPLOTLIB_AVAILABLE_PAGE:
                st.pyplot(fig_heatmap)
                plt.close(fig_heatmap)
    
    with col6:
        st.write("**Correlações Principais**")
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                corr_pairs.append((var1, var2, corr_val))
        
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        for var1, var2, corr in corr_pairs[:6]:
            strength = "forte" if abs(corr) > 0.5 else "moderada" if abs(corr) > 0.3 else "fraca"
            direction = "positiva" if corr > 0 else "negativa"
            emoji = "🔴" if abs(corr) > 0.5 else "🟡" if abs(corr) > 0.3 else "⚪"
            st.write(f"{emoji} **{var1}** vs **{var2}**: {corr:.3f} ({strength} {direction})")
        
        csv_corr = corr_matrix.to_csv()
        st.download_button(
            label="📥 Download Matriz de Correlação",
            data=csv_corr,
            file_name="correlation_matrix.csv",
            mime="text/csv"
        )

st.markdown("---")
st.caption("Análises Avançadas - Dashboard de QA Analysis")
