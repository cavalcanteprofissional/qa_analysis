"""Cálculo de métricas e geração de relatórios com overlap palavra-resposta-contexto."""
from typing import Dict, Any
from pathlib import Path
import pandas as pd
import json


class MetricsCalculator:
    @staticmethod
    def _compute_word_overlap(answer_str: str, context_str: str) -> tuple:
        """Calcula overlap de palavras entre resposta e contexto.
        
        Retorna (overlap_count, overlap_fraction) onde:
        - overlap_count: número de palavras da resposta que aparecem no contexto
        - overlap_fraction: overlap_count / total de palavras na resposta (0 se resposta vazia)
        """
        if pd.isna(answer_str) or pd.isna(context_str):
            return 0, 0.0
        
        # normaliza strings
        answer_words = set(str(answer_str).lower().split())
        context_words = set(str(context_str).lower().split())
        
        if not answer_words:
            return 0, 0.0
        
        overlap_count = len(answer_words & context_words)
        overlap_fraction = overlap_count / len(answer_words)
        
        return overlap_count, overlap_fraction
    
    def annotate_overlap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Anota DataFrame com overlap de palavras (resposta vs contexto) por linha.

        overlap_count: número de palavras da resposta que aparecem no contexto.
        overlap_fraction: overlap_count / total de palavras na resposta (0 se resposta vazia).
        Respostas faltantes (NaN) recebem overlap_count 0 e overlap_fraction 0.0.
        """
        if df.empty:
            return df.copy()

        df = df.copy()
        if "answer" not in df.columns:
            df["answer"] = pd.NA
        if "context" not in df.columns:
            return df

        # calcula overlap palavra-a-palavra para cada linha
        overlaps = df.apply(
            lambda row: pd.Series(
                self._compute_word_overlap(row.get("answer"), row.get("context")),
                index=["overlap_count", "overlap_fraction"]
            ),
            axis=1
        )
        
        df["overlap_count"] = overlaps["overlap_count"].astype(int)
        df["overlap_fraction"] = overlaps["overlap_fraction"].astype(float)
        
        return df

    def calculate_all_metrics(self, df_results: pd.DataFrame) -> Dict[str, Any]:
        # ensure overlap annotation present
        df_with_overlap = self.annotate_overlap(df_results) if "overlap_count" not in df_results.columns else df_results.copy()

        metrics = {
            "overall": self._calculate_overall_metrics(df_with_overlap),
            "per_model": self._calculate_per_model_metrics(df_with_overlap),
            "comparative": self._calculate_comparative_metrics(df_with_overlap),
            "categorical": self._categorize_responses(df_with_overlap),
        }
        return metrics

    def _calculate_overall_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        total = len(df)
        avg_overlap_fraction = float(df["overlap_fraction"].mean()) if "overlap_fraction" in df.columns and len(df) > 0 else 0.0
        avg_overlap_count = float(df["overlap_count"].mean()) if "overlap_count" in df.columns and len(df) > 0 else 0.0
        return {"total_predictions": int(total), "avg_overlap_fraction": avg_overlap_fraction, "avg_overlap_count": avg_overlap_count}

    def _calculate_per_model_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        out = {}
        for model, g in df.groupby("model"):
            scores = g["score"].dropna().astype(float)
            avg_overlap_fraction = float(g["overlap_fraction"].mean()) if "overlap_fraction" in g.columns and len(g) > 0 else 0.0
            avg_overlap_count = float(g["overlap_count"].mean()) if "overlap_count" in g.columns and len(g) > 0 else 0.0
            out[model] = {
                "count": int(len(g)),
                "mean_score": float(scores.mean()) if not scores.empty else None,
                "median_score": float(scores.median()) if not scores.empty else None,
                "avg_overlap_fraction": avg_overlap_fraction,
                "avg_overlap_count": avg_overlap_count,
            }
        return out

    def _calculate_comparative_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        out = {}
        grouped = df.groupby(["question", "context"])
        overlaps = []
        for _, g in grouped:
            answers = g["answer"].astype(str).tolist()
            unique = len(set(answers))
            overlaps.append({"n_models": len(answers), "n_unique_answers": unique})

        out["avg_unique_answers"] = float(pd.DataFrame(overlaps).n_unique_answers.mean()) if overlaps else 0.0
        return out

    def _categorize_responses(self, df: pd.DataFrame) -> Dict[str, Any]:
        def cat(score: float) -> str:
            if score is None:
                return "unknown"
            if score >= 0.8:
                return "low_risk"
            if score >= 0.5:
                return "medium_risk"
            return "high_risk"

        cats = df["score"].fillna(0.0).astype(float).apply(cat)
        return cats.value_counts().to_dict()

    def generate_report(self, metrics: Dict, output_path: Path) -> None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # JSON
        json_path = output_path / "metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        # Markdown summary
        md_path = output_path / "metrics_summary.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Metrics Summary\n\n")
            f.write("## Overall\n")
            for k, v in metrics.get("overall", {}).items():
                f.write(f"- {k}: {v}\n")
            f.write("\n## Per Model\n")
            for m, stats in metrics.get("per_model", {}).items():
                f.write(f"### {m}\n")
                for kk, vv in stats.items():
                    f.write(f"- {kk}: {vv}\n")

        # CSV: flatten per_model
        per_model = metrics.get("per_model", {})
        rows = []
        for m, s in per_model.items():
            row = {"model": m}
            row.update(s)
            rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(output_path / "per_model_metrics.csv", index=False)
