"""Controlador da pipeline que orquestra carregamento, seleção e execução."""
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import torch
import pandas as pd

from .data_loader import ShardLoader
from .model_selector import ModelSelector
from .parallel_processor import ParallelProcessor
from .metrics_calculator import MetricsCalculator
from .logger_config import setup_logging


class PipelineController:
    def __init__(self,
                 shards: List[str] = None,
                 models: List[str] = None,
                 batch_size: int = 32,
                 workers: int = None,
                 output_dir: str = "outputs",
                 log_dir: str = "logs") -> None:
        self.logger = setup_logging(log_dir)
        self.shard_loader = ShardLoader("data/shards")
        self.model_selector = ModelSelector()
        self.parallel = ParallelProcessor(max_workers=workers)
        self.metrics = MetricsCalculator()
        self.shards = shards
        self.models = models
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, max_samples: int = None, export_formats: List[str] = None) -> Dict[str, Any]:
        """Executa toda a pipeline e retorna o dicionário de resultados.

        1. Seleciona e carrega shards
        2. Seleciona modelos (descritores)
        3. Executa processamento paralelo
        4. Gera métricas e exporta relatórios
        """
        self.logger.info("Starting pipeline run")

        # 1. Load data
        df = self.shard_loader.load_selected_shards(self.shards)
        if df.empty:
            self.logger.warning("No data loaded for selected shards")
            return {}

        # optionally limit samples
        if max_samples:
            df = df.head(max_samples)

        # Ensure required columns; allow common alternatives
        if "question" not in df.columns or "context" not in df.columns:
            # handle common shard schema: 'query' -> question, 'text' -> context
            if "query" in df.columns and "text" in df.columns:
                self.logger.info("Mapping input columns: 'query'->'question', 'text'->'context'")
                df = df.rename(columns={"query": "question", "text": "context"})
            else:
                self.logger.error("Input data must contain 'question' and 'context' columns (or 'query' and 'text')")
                return {}

        # 2. select models
        model_descriptors = self.model_selector.select_models(self.models or ["all"])
        if not model_descriptors:
            self.logger.error("No models selected")
            return {}

        # 3. parallel processing
        use_cuda = torch.cuda.is_available()
        self.logger.info(f"CUDA available: {use_cuda}")
        results = self.parallel.process_models_parallel(model_descriptors, df, batch_size=self.batch_size, use_cuda=use_cuda)

        # 4. aggregate results into unified DataFrame
        out_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)

        all_rows = []
        for model_key, res in results.items():
            if isinstance(res, dict) and res.get("error"):
                self.logger.error(f"Model {model_key} failed: {res.get('error')}")
                continue
            for r in res:
                r_out = r.copy()
                r_out["model"] = model_key
                all_rows.append(r_out)

        if not all_rows:
            self.logger.warning("No results produced by models")
            return results

        df_results = pd.DataFrame(all_rows)

        # annotate overlaps before saving consolidated CSV
        try:
            df_results = self.metrics.annotate_overlap(df_results)
        except Exception:
            self.logger.warning("Could not annotate overlap on results; continuing without overlap")

        # save consolidated CSV
        csv_path = out_dir / "results_consolidated.csv"
        df_results.to_csv(csv_path, index=False)
        self.logger.info(f"Saved consolidated results to {csv_path}")

        # 5. metrics and reports
        metrics = self.metrics.calculate_all_metrics(df_results)
        self.metrics.generate_report(metrics, out_dir)

        return {"results_df": df_results, "metrics": metrics, "out_dir": out_dir}
