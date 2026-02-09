import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# Status flags for dependencies
PLOTLY_AVAILABLE = False
MATPLOTLIB_AVAILABLE = False

# Try to import Plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Try to import Matplotlib/Seaborn
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for Streamlit
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    st.write("<!-- Matplotlib not available -->")

# Try to import MetricsCalculator
try:
    from src.metrics_calculator import MetricsCalculator
    _METRICS = MetricsCalculator()
except Exception:
    _METRICS = None

# Try to import ColorManager
try:
    from src.color_manager import ColorManager
    _COLOR_MANAGER_AVAILABLE = True
except ImportError:
    _COLOR_MANAGER_AVAILABLE = False


def find_latest_results_csv(outputs_dir: Path = Path("outputs")) -> Optional[Path]:
    """Find the latest results_consolidated.csv file."""
    if not outputs_dir.exists():
        return None

    # Try to find directories with timestamp pattern YYYYMMDD_HHMMSS
    pattern = re.compile(r"^\d{8}_\d{6}$")
    cand_dirs = [p for p in outputs_dir.iterdir() if p.is_dir() and pattern.match(p.name)]
    if cand_dirs:
        latest = max(cand_dirs, key=lambda p: p.name)
        candidate = latest / "results_consolidated.csv"
        if candidate.exists():
            return candidate

    # Fallback: pick newest directory by mtime
    dirs = [p for p in outputs_dir.iterdir() if p.is_dir()]
    if dirs:
        latest = max(dirs, key=lambda p: p.stat().st_mtime)
        candidate = latest / "results_consolidated.csv"
        if candidate.exists():
            return candidate

    # As last resort search recursively for any results_consolidated.csv and pick newest
    matches = list(outputs_dir.rglob("results_consolidated.csv"))
    if matches:
        return max(matches, key=lambda p: p.stat().st_mtime)

    return None


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV file with caching."""
    df = pd.read_csv(path)
    return df


def map_columns(cols):
    """Map column names to standard names."""
    mapping = {
        "query": ["question", "query", "prompt"],
        "answer": ["answer", "prediction", "generated_answer"],
        "context": ["context", "passage", "source_text"],
        "score": ["score", "model_score", "confidence"],
        "overlap": ["overlap", "word_overlap", "token_overlap"],
        "model": ["model", "model_name"],
    }
    found = {}
    lower = {c.lower(): c for c in cols}
    for key, candidates in mapping.items():
        for cand in candidates:
            if cand in lower:
                found[key] = lower[cand]
                break
    return found


def compute_word_overlap(answer: str, context: str) -> float:
    """Compute word overlap between answer and context."""
    if not answer or pd.isna(answer):
        return 0.0
    aw = set(str(answer).lower().split())
    cw = set(str(context).lower().split())
    if not aw:
        return 0.0
    return float(len(aw & cw) / len(aw))


def create_fallback_visualization():
    """Show fallback message when visualization libraries are not available."""
    st.error("🚫 **Bibliotecas de visualização não disponíveis**")
    st.info("📋 **Para visualizações interativas, instale:**")
    st.code("pip install plotly matplotlib seaborn")
    st.write("**Dados disponíveis em formato de tabela abaixo:**")


def create_histogram(data, x_col, title, nbins=40):
    """Create histogram with fallback."""
    if PLOTLY_AVAILABLE:
        return px.histogram(data, x=x_col, nbins=nbins, title=title)
    elif MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(data[x_col].dropna(), bins=nbins, alpha=0.7, edgecolor='black')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    return None


def create_scatter(data, x_col, y_col, color_col=None, title="", color_manager=None):
    """Create scatter plot with fallback and dynamic color management."""
    if PLOTLY_AVAILABLE:
        if color_col and color_col in data.columns and color_manager:
            # Apply consistent colors using ColorManager
            color_discrete_map = {}
            for model in data[color_col].unique():
                color_discrete_map[model] = color_manager.get_model_color(model)
            return px.scatter(data, x=x_col, y=y_col, color=color_col, 
                            color_discrete_map=color_discrete_map, title=title)
        return px.scatter(data, x=x_col, y=y_col, color=color_col, title=title)
    elif MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if color_col and color_col in data.columns:
            categories = data[color_col].unique()
            
            # Use ColorManager if available, otherwise fallback to tab10
            if color_manager and _COLOR_MANAGER_AVAILABLE:
                for i, cat in enumerate(categories):
                    subset = data[data[color_col] == cat]
                    color = color_manager.get_model_color(cat)
                    ax.scatter(subset[x_col], subset[y_col], 
                              c=[color], label=cat, alpha=0.7, s=50)
            else:
                colors = plt.cm.tab10(range(len(categories)))
                for i, cat in enumerate(categories):
                    subset = data[data[color_col] == cat]
                    ax.scatter(subset[x_col], subset[y_col], 
                              c=[colors[i]], label=cat, alpha=0.7, s=50)
            ax.legend()
        else:
            ax.scatter(data[x_col], data[y_col], alpha=0.7, s=50)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    return None


def create_bar(data, x_col, y_col, title="", color_manager=None):
    """Create bar chart with fallback and model color differentiation."""
    if PLOTLY_AVAILABLE:
        if color_manager and x_col in data.columns:
            # Apply model-specific colors to bars
            color_discrete_map = {}
            for model in data[x_col].unique():
                color_discrete_map[model] = color_manager.get_model_color(model)
            return px.bar(data, x=x_col, y=y_col, color=x_col,
                         color_discrete_map=color_discrete_map, title=title)
        return px.bar(data, x=x_col, y=y_col, title=title)
    elif MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if color_manager and x_col in data.columns and _COLOR_MANAGER_AVAILABLE:
            # Apply model-specific colors
            colors = [color_manager.get_model_color(model) for model in data[x_col]]
            bars = ax.bar(data[x_col], data[y_col], color=colors, alpha=0.7)
        else:
            bars = ax.bar(data[x_col], data[y_col], alpha=0.7)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    return None


def display_chart(fig, plotly_func, *args, **kwargs):
    """Display chart with proper backend."""
    if PLOTLY_AVAILABLE and fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        return True
    
    fig = plotly_func(*args, **kwargs)
    if MATPLOTLIB_AVAILABLE and fig is not None:
        st.pyplot(fig)
        plt.close(fig)
        return True
    
    create_fallback_visualization()
    return False


def main():
    """Main function for Streamlit app."""
    st.set_page_config(page_title="QA Analysis Dashboard", layout="wide")
    st.title("📊 QA Analysis Dashboard")
    
    # Initialize ColorManager
    color_manager = None
    if _COLOR_MANAGER_AVAILABLE:
        color_manager = ColorManager()
        color_manager.load_from_session_state()
    csv_path = find_latest_results_csv()
    if not csv_path:
        st.error("❌ Nenhum arquivo results_consolidated.csv encontrado em outputs/.")
        st.info("💡 Execute o pipeline QA primeiro para gerar dados de análise.")
        return

    st.sidebar.markdown(f"📁 **CSV carregado:** `{csv_path.name}`")

    try:
        with st.spinner("📊 Carregando dados..."):
            df = load_csv(csv_path)
    except Exception as e:
        st.error(f"❌ Erro ao ler CSV: {e}")
        return

    if df.empty:
        st.warning("⚠️ CSV carregado, mas está vazio.")
        return

    cols = list(df.columns)
    colmap = map_columns(cols)

    # Ensure basic columns
    qcol = colmap.get("query") or ("question" if "question" in cols else None)
    acol = colmap.get("answer") or ("answer" if "answer" in cols else None)
    ccol = colmap.get("context") or ("context" if "context" in cols else None)
    scol = colmap.get("score") or ("score" if "score" in cols else None)
    ocol = colmap.get("overlap") or ("overlap" if "overlap" in cols else None)
    mcol = colmap.get("model") or ("model" if "model" in cols else None)

    missing = [k for k, v in {"question": qcol, "answer": acol, "context": ccol, "score": scol}.items() if v is None]
    if missing:
        st.error(f"❌ Colunas obrigatórias ausentes no CSV: {', '.join(missing)}")
        st.write("📋 **Colunas encontradas:**", cols)
        return

    # If overlap missing, annotate using available function
    if ocol is None:
        st.info("🔧 Calculando overlap palavra-contexto...")
        if _METRICS is not None:
            df = _METRICS.annotate_overlap(df)
            ocol = "overlap"
        else:
            df["overlap"] = df.apply(lambda r: compute_word_overlap(r.get(acol), r.get(ccol)), axis=1).astype(float)
            ocol = "overlap"

    # Normalize numeric columns
    df[scol] = pd.to_numeric(df[scol], errors="coerce")
    df[ocol] = pd.to_numeric(df[ocol], errors="coerce")

# Compute question length in words (for all rows)
    df["question_length_words"] = df[qcol].fillna("").apply(lambda x: len(str(x).split()))
    
    # Compute question length in characters (excluding spaces)
    df["question_length_chars"] = df[qcol].fillna("").apply(lambda x: len(str(x).replace(" ", "")))

# Main metrics
    total_rows = len(df)
    mean_q_len = float(df["question_length_words"].mean())
    mean_q_len_chars = float(df["question_length_chars"].mean())
    mean_score = float(df[scol].mean()) if not df[scol].isna().all() else None
    mean_overlap = float(df[ocol].mean()) if not df[ocol].isna().all() else None

# Top-level metrics display
    c1, c2, c3, c4, c5 = st.columns([1.6, 1.6, 1.6, 1.6, 3])
    c1.metric("Total linhas", f"{total_rows}")
    c2.metric("Tamanho Médio das Querys (palavra)", f"{mean_q_len:.2f}")
    c3.metric("Tamanho Médio das Querys (letra)", f"{mean_q_len_chars:.2f}")
    c4.metric("Score médio", f"{mean_score:.4f}" if mean_score is not None else "N/A")
    c5.metric("Overlap médio", f"{mean_overlap:.4f}" if mean_overlap is not None else "N/A")

    # Interpretation
    with st.expander("📖 Interpretação de Overlap"):
        st.write("- **Alta sobreposição** → resposta provavelmente explícita no contexto.")
        st.write("- **Baixa sobreposição** → possível inferência incorreta ou alucinação.")

# Sidebar filters
    st.sidebar.header("🔍 Filtros")
    min_score = st.sidebar.slider("Score mínimo", 0.0, 1.0, 0.0, 0.01)
    min_overlap = st.sidebar.slider("Overlap mínimo", 0.0, 1.0, 0.0, 0.01)
    models = sorted(df[mcol].unique().tolist()) if mcol in df.columns else ["(single)"]
    selected_models = st.sidebar.multiselect("Modelos", options=models, default=models)
    keyword = st.sidebar.text_input("Buscar palavra-chave em pergunta/resposta")

    # Color Management Sidebar
    if color_manager and _COLOR_MANAGER_AVAILABLE:
        st.sidebar.markdown("---")
        st.sidebar.header("🎨 Gerenciamento de Cores")

        # Palette Selection
        palette_options = color_manager.get_all_palette_options()
        try:
            current_index = palette_options.index(color_manager.current_palette)
        except ValueError:
            current_index = 0
            
        selected_palette = st.sidebar.selectbox(
            "Tipo de Paleta",
            options=palette_options,
            index=current_index
        )

        if selected_palette != color_manager.current_palette:
            color_manager.set_palette(selected_palette)

        # Model Color Assignment
        if mcol in df.columns:
            st.sidebar.subheader("🎯 Cores dos Modelos")
            
            for model in sorted(models):
                current_color = color_manager.get_model_color(model)
                color_hex = st.sidebar.color_picker(
                    f"Cor: {model}",
                    value=current_color,
                    key=f"color_{model}"
                )
                if color_hex != current_color:
                    color_manager.update_model_color(model, color_hex)

        # Advanced Options
        st.sidebar.subheader("⚙️ Opções Avançadas")
        
        # Get model performance stats for performance coloring
        model_performance = {}
        if mcol in df.columns and scol in df.columns:
            model_performance = color_manager.get_model_performance_stats(df, mcol, scol)
        
        performance_mode = st.sidebar.checkbox(
            "Colorir por Performance (verde=alto, vermelho=baixo)",
            value=color_manager.performance_mode,
            disabled=len(model_performance) == 0
        )

        if performance_mode != color_manager.performance_mode:
            color_manager.set_performance_mode(performance_mode)

        accessibility_mode = st.sidebar.checkbox(
            "Modo Acessibilidade",
            value=color_manager.accessibility_mode
        )

        if accessibility_mode != color_manager.accessibility_mode:
            color_manager.set_accessibility_mode(accessibility_mode)

        # Show performance mapping if performance mode is enabled
        if performance_mode and model_performance:
            st.sidebar.write("**Mapeamento de Performance:**")
            st.sidebar.write("• 🟢 Excelente (≥0.8)")
            st.sidebar.write("• 🟠 Bom (0.6-0.79)")
            st.sidebar.write("• 🔴 Médio (0.4-0.59)")
            st.sidebar.write("• 🟤 Ruim (<0.4)")

        # Custom Palette Creation
        st.sidebar.subheader("🔧 Paleta Personalizada")
        custom_palette_name = st.sidebar.text_input("Nome da nova paleta")

        if custom_palette_name and st.sidebar.button("💾 Salvar Paleta Atual"):
            if mcol in df.columns:
                current_colors = [color_manager.get_model_color(m) for m in sorted(models)]
                color_manager.save_custom_palette(custom_palette_name, current_colors)
                st.sidebar.success(f"Paleta '{custom_palette_name}' salva!")

        # Reset Button
        if st.sidebar.button("🔄 Redefinir Cores"):
            color_manager.reset_colors()
            st.rerun()

        # Color Preview
        if mcol in df.columns:
            st.sidebar.subheader("📋 Prévia de Cores")
            for model in sorted(models):
                color = color_manager.get_model_color(model)
                st.sidebar.markdown(f'<div style="display: flex; align-items: center;">'
                                   f'<div style="width: 20px; height: 20px; background-color: {color}; '
                                   f'border: 1px solid #ccc; margin-right: 10px;"></div>'
                                   f'<span>{model}</span></div>', 
                                   unsafe_allow_html=True)

    # Apply filters
    filt = df[df[scol].fillna(0) >= min_score]
    filt = filt[filt[ocol].fillna(0) >= min_overlap]
    if mcol in df.columns:
        filt = filt[filt[mcol].isin(selected_models)]
    if keyword:
        kw = keyword.lower()
        filt = filt[filt[qcol].fillna("").str.lower().str.contains(kw) | filt[acol].fillna("").str.lower().str.contains(kw)]

    st.subheader("📈 Visualizações")
    
    # Visualization 1: Score Distribution
    fig1 = create_histogram(filt, scol, "Distribuição de Scores", nbins=40)
    display_chart(fig1, create_histogram, filt, scol, "Distribuição de Scores", nbins=40)

# Visualizations 2 & 3: Two-column layout
    colA, colB = st.columns(2)
    
    with colA:
        st.write("**Score vs Overlap:**")
        fig2 = create_scatter(filt, ocol, scol,
                             color_col=mcol if mcol in df.columns else None, 
                             title="Score vs Overlap",
                             color_manager=color_manager)
        if PLOTLY_AVAILABLE and fig2 is not None:
            st.plotly_chart(fig2, use_container_width=True)
        elif MATPLOTLIB_AVAILABLE and fig2 is not None:
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.write("Correlação não disponível")
    
    with colB:
        st.write("**Estatísticas por Modelo:**")
        if mcol in df.columns:
            model_stats = filt.groupby(mcol).agg({
                scol: ['mean', 'count', 'std'],
                ocol: 'mean'
            }).round(4)
            model_stats.columns = ['Score Médio', 'Total', 'Score Std', 'Overlap Médio']
            st.dataframe(model_stats)
        else:
            st.write("Análise por modelo não disponível")

    # Visualization 4: Question Length by Model
    qlen_by_model = None
    if mcol in df.columns:
        st.write("**Tamanho médio das perguntas por modelo:**")
        qlen_by_model = filt.groupby(mcol)["question_length_words"].mean().reset_index()
        
        fig3 = create_bar(qlen_by_model, mcol, "question_length_words", 
                          "Tamanho médio das perguntas por modelo (palavras)",
                          color_manager=color_manager)
        if PLOTLY_AVAILABLE and fig3 is not None:
            st.plotly_chart(fig3, use_container_width=True)
        elif MATPLOTLIB_AVAILABLE and fig3 is not None:
            st.pyplot(fig3)
            plt.close(fig3)
        else:
            st.dataframe(qlen_by_model)

    # Examples: top 10, bottom 10
    st.subheader("🏆 Exemplos destacados")
    top10 = df.sort_values(by=scol, ascending=False).head(10)
    bot10 = df.sort_values(by=scol, ascending=True).head(10)

    with st.expander("🔝 Top 10 por score"):
        display_cols = [qcol, acol, ccol, scol, ocol] + ([mcol] if mcol in df.columns else [])
        st.dataframe(top10[display_cols], height=400)

    with st.expander("🔻 Bottom 10 por score"):
        display_cols = [qcol, acol, ccol, scol, ocol] + ([mcol] if mcol in df.columns else [])
        st.dataframe(bot10[display_cols], height=400)

# Examples with random selection showing all model responses side-by-side
    st.subheader("🔄 Exemplos com respostas de todos os modelos")
    if mcol in df.columns:
        # Exclude rows present in top/bottom lists by index
        excluded_indices = set(top10.index.tolist() + bot10.index.tolist())
        available_data = df[~df.index.isin(excluded_indices)]
        
        # Helper functions for disagreement classification
        def count_unique_answers(group_data, answer_col):
            """Count unique answers, treating same text with different scores as same."""
            answers = group_data[answer_col].fillna("").astype(str).tolist()
            return len(set(answers))
        
        def classify_disagreement_level(group_data, model_col, answer_col):
            """Classify the level of disagreement in a group."""
            models = group_data[model_col].unique()
            
            # Require at least 2 models
            if len(models) < 2:
                return "insufficient_models"
            
            unique_answers = count_unique_answers(group_data, answer_col)
            
            if unique_answers == len(models):
                return "full_disagreement"  # All models have different answers
            elif unique_answers > 1:
                return "partial_disagreement"  # Some models agree, but not all
            else:
                return "no_disagreement"  # All models give same answer
        
        # Get unique question-context pairs
        unique_pairs = available_data[[qcol, ccol]].drop_duplicates()
        
        # Tier 1: Find examples with full disagreement (all models different)
        full_disagreement_pairs = []
        partial_disagreement_pairs = []
        
        for _, pair in unique_pairs.iterrows():
            question = pair[qcol]
            context = pair[ccol]
            
            # Get all model responses for this question-context pair
            example_data = df[(df[qcol] == question) & (df[ccol] == context)]
            
            # Classify disagreement level
            disagreement_level = classify_disagreement_level(example_data, mcol, acol)
            
            if disagreement_level == "full_disagreement":
                full_disagreement_pairs.append(pair)
            elif disagreement_level == "partial_disagreement":
                partial_disagreement_pairs.append(pair)
        
        # Convert to DataFrames for sampling
        df_full_disagreement = pd.DataFrame(full_disagreement_pairs)
        df_partial_disagreement = pd.DataFrame(partial_disagreement_pairs)
        
        # Tiered selection with fallback
        selected_pairs = []
        
        # First, try to get examples from full disagreement
        if len(df_full_disagreement) > 0:
            n_from_full = min(5, len(df_full_disagreement))
            selected_from_full = df_full_disagreement.sample(n=n_from_full, random_state=42)
            selected_pairs.extend(selected_from_full.to_dict('records'))
        
        # If we need more examples, add from partial disagreement
        if len(selected_pairs) < 5 and len(df_partial_disagreement) > 0:
            remaining_needed = 5 - len(selected_pairs)
            n_from_partial = min(remaining_needed, len(df_partial_disagreement))
            selected_from_partial = df_partial_disagreement.sample(n=n_from_partial, random_state=42)
            selected_pairs.extend(selected_from_partial.to_dict('records'))
        
        # Convert final selection to DataFrame
        if selected_pairs:
            selected_pairs_df = pd.DataFrame(selected_pairs)
            # Shuffle the final selection to mix full and partial disagreement examples
            selected_pairs_df = selected_pairs_df.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            selected_pairs_df = pd.DataFrame()  # Empty DataFrame
        
        if len(selected_pairs_df) > 0:
            # Display each selected example in side-by-side format
            for idx, (_, pair) in enumerate(selected_pairs_df.iterrows()):
                st.subheader(f"Exemplo {idx + 1}")
                
                question = pair[qcol]
                context = pair[ccol]
                
                # Get all model responses for this question-context pair
                example_data = df[(df[qcol] == question) & (df[ccol] == context)]
                
                # Classify disagreement level for display
                disagreement_level = classify_disagreement_level(example_data, mcol, acol)
                
                # Display question and context (shared by all models)
                st.write("**Pergunta:**", question)
                st.write("**Contexto:**", context[:200] + "..." if len(str(context)) > 200 else context)
                
                # Create side-by-side columns for each model
                models = sorted(example_data[mcol].unique())
                cols = st.columns(len(models))
                
                for i, model in enumerate(models):
                    with cols[i]:
                        model_data = example_data[example_data[mcol] == model].iloc[0]
                        st.markdown(f"### {model}")
                        st.write("**Resposta:**", model_data[acol])
                        st.write("**Score:**", f"{model_data[scol]:.3f}")
                        st.write("**Overlap:**", f"{model_data[ocol]:.3f}")
                
                st.divider()  # Add separator between examples
        else:
            st.write("Nenhum exemplo com divergência entre modelos encontrado (todos foram excluídos pelos top/bottom ou não há divergência suficiente entre modelos).")
    else:
        st.write("Dataset contém apenas um modelo; não há comparação entre modelos.")

    # Show filtered table overview
    with st.expander("📋 Tabela filtrada (visualizar)"):
        st.dataframe(filt.reset_index(drop=True), height=600)


if __name__ == "__main__":
    main()