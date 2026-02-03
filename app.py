import re
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from src.metrics_calculator import MetricsCalculator
    _METRICS = MetricsCalculator()
except Exception:
    _METRICS = None


def find_latest_results_csv(outputs_dir: Path = Path("outputs")) -> Optional[Path]:
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
    df = pd.read_csv(path)
    return df


def map_columns(cols):
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
    if not answer or pd.isna(answer):
        return 0.0
    aw = set(str(answer).lower().split())
    cw = set(str(context).lower().split())
    if not aw:
        return 0.0
    return float(len(aw & cw) / len(aw))


def main():
    st.set_page_config(page_title="QA Analisys Dashboard", layout="wide")
    st.title("QA Analisys Dashboard")

    csv_path = find_latest_results_csv()
    if not csv_path:
        st.error("Nenhum arquivo results_consolidated.csv encontrado em outputs/.")
        return

    st.sidebar.markdown(f"**CSV carregado:** {csv_path}")

    try:
        df = load_csv(csv_path)
    except Exception as e:
        st.error(f"Erro ao ler CSV: {e}")
        return

    if df.empty:
        st.warning("CSV carregado, mas está vazio.")
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
        st.error(f"Colunas obrigatórias ausentes no CSV: {', '.join(missing)}")
        return

    # If overlap missing, annotate using available function
    if ocol is None:
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
    mean_score = float(df[scol].mean()) if df[scol].notna().any() else None
    mean_overlap = float(df[ocol].mean()) if df[ocol].notna().any() else None

    # Top-level metrics display
    c1, c2, c3, c4 = st.columns([2, 2, 2, 3])
    c1.metric("Total linhas", f"{total_rows}")
    c2.metric("Tamanho médio (palavras)", f"{mean_q_len:.2f}")
    c3.metric("Score médio", f"{mean_score:.4f}" if mean_score is not None else "N/A")
    c4.metric("Overlap médio", f"{mean_overlap:.4f}" if mean_overlap is not None else "N/A")

    # Interpretation
    with st.expander("Interpretação de Overlap"):
        st.write("- Alta sobreposição → resposta provavelmente explícita no contexto.")
        st.write("- Baixa sobreposição → possível inferência incorreta ou alucinação.")

    # Sidebar filters
    st.sidebar.header("Filtros")
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

    st.subheader("Visualizações")
    fig1 = px.histogram(filt, x=scol, nbins=40, title="Distribuição de Scores")
    fig2 = px.scatter(filt, x=ocol, y=scol, color=mcol if mcol in df.columns else None, title="Score vs Overlap")
    qlen_by_model = None
    if mcol in df.columns:
        qlen_by_model = filt.groupby(mcol)["_q_len_words"].mean().reset_index()
        fig3 = px.bar(qlen_by_model, x=mcol, y="_q_len_words", title="Tamanho médio das perguntas por modelo (palavras)")

    cA, cB = st.columns(2)
    cA.plotly_chart(fig1, use_container_width=True)
    cB.plotly_chart(fig2, use_container_width=True)
    if qlen_by_model is not None:
        st.plotly_chart(fig3, use_container_width=True)

    # Examples: top 10, bottom 10
    st.subheader("Exemplos destacados")
    top10 = df.sort_values(by=scol, ascending=False).head(10)
    bot10 = df.sort_values(by=scol, ascending=True).head(10)

    with st.expander("Top 10 por score"):
        display_cols = [qcol, acol, ccol, scol, ocol] + ([mcol] if mcol in df.columns else [])
        st.dataframe(top10[display_cols], height=400)

    with st.expander("Bottom 10 por score"):
        display_cols = [qcol, acol, ccol, scol, ocol] + ([mcol] if mcol in df.columns else [])
        st.dataframe(bot10[display_cols], height=400)

    # Examples where models disagree (different answers for same question/context)
    st.subheader("Exemplos com respostas divergentes entre modelos")
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
    with st.expander("Tabela filtrada (visualizar)"):
        st.dataframe(filt.reset_index(drop=True), height=600)


if __name__ == "__main__":
    main()
