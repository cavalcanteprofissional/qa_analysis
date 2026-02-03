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
    st.write("<!-- Plotly successfully imported -->")
except ImportError:
    st.write("<!-- Plotly not available -->")

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


def create_scatter(data, x_col, y_col, color_col=None, title=""):
    """Create scatter plot with fallback."""
    if PLOTLY_AVAILABLE:
        return px.scatter(data, x=x_col, y=y_col, color=color_col, title=title)
    elif MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if color_col and color_col in data.columns:
            categories = data[color_col].unique()
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


def create_bar(data, x_col, y_col, title=""):
    """Create bar chart with fallback."""
    if PLOTLY_AVAILABLE:
        return px.bar(data, x=x_col, y=y_col, title=title)
    elif MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(10, 6))
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
    
    # Show dependency status
    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        if PLOTLY_AVAILABLE:
            st.success("✅ Plotly Available")
        else:
            st.error("❌ Plotly Not Available")
    
    with status_col2:
        if MATPLOTLIB_AVAILABLE:
            st.success("✅ Matplotlib Available")
        else:
            st.error("❌ Matplotlib Not Available")
    
    with status_col3:
        if _METRICS is not None:
            st.success("✅ Metrics Calculator Available")
        else:
            st.warning("⚠️ Metrics Calculator Fallback")

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

    st.success(f"✅ **Dados carregados:** {len(df)} linhas, {len(df.columns)} colunas")

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
    df["_q_len_words"] = df[qcol].fillna("").apply(lambda x: len(str(x).split()))

    # Main metrics
    total_rows = len(df)
    mean_q_len = float(df["_q_len_words"].mean())
    mean_score = float(df[scol].mean()) if not df[scol].isna().all() else None
    mean_overlap = float(df[ocol].mean()) if not df[ocol].isna().all() else None

    # Top-level metrics display
    c1, c2, c3, c4 = st.columns([2, 2, 2, 3])
    c1.metric("Total linhas", f"{total_rows}")
    c2.metric("Tamanho médio (palavras)", f"{mean_q_len:.2f}")
    c3.metric("Score médio", f"{mean_score:.4f}" if mean_score is not None else "N/A")
    c4.metric("Overlap médio", f"{mean_overlap:.4f}" if mean_overlap is not None else "N/A")

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

    # Apply filters
    filt = df[df[scol].fillna(0) >= min_score]
    filt = filt[filt[ocol].fillna(0) >= min_overlap]
    if mcol in df.columns:
        filt = filt[filt[mcol].isin(selected_models)]
    if keyword:
        kw = keyword.lower()
        filt = filt[filt[qcol].fillna("").str.lower().str.contains(kw) | filt[acol].fillna("").str.lower().str.contains(kw)]

    st.info(f"📊 **Dados filtrados:** {len(filt)} linhas ({len(filt)/len(df)*100:.1f}% do total)")

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
                            title="Score vs Overlap")
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
        qlen_by_model = filt.groupby(mcol)["_q_len_words"].mean().reset_index()
        
        fig3 = create_bar(qlen_by_model, mcol, "_q_len_words", 
                         "Tamanho médio das perguntas por modelo (palavras)")
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

    # Examples where models disagree (different answers for same question/context)
    st.subheader("🔄 Exemplos com respostas divergentes entre modelos")
    if mcol in df.columns:
        grouped = df.groupby([qcol, ccol])
        discord = []
        for _, g in grouped:
            answers = g[acol].fillna("").astype(str).tolist()
            if len(set(answers)) > 1 and len(g) > 1:
                discord.extend(g.to_dict("records"))
        # Exclude rows present in top/bottom lists by index
        excluded_indices = set(top10.index.tolist() + bot10.index.tolist())
        discord_filtered = [r for r in discord if r.get("_id") not in excluded_indices][:5]
        if discord_filtered:
            df_discord = pd.DataFrame(discord_filtered)
            display_cols = [qcol, acol, ccol, scol, ocol, mcol]
            st.dataframe(df_discord[display_cols], height=400)
        else:
            st.write("Nenhum exemplo divergente encontrado (ou foram excluídos pelos top/bottom).")
    else:
        st.write("Dataset contém apenas um modelo; não há comparação entre modelos.")

    # Show filtered table overview
    with st.expander("📋 Tabela filtrada (visualizar)"):
        st.dataframe(filt.reset_index(drop=True), height=600)


if __name__ == "__main__":
    main()