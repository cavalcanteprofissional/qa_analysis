"""Processamento paralelo de modelos.

Executa a inferência em processos separados. Cada processo instancia o
pipeline HF localmente para evitar problemas de pickling.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any
import multiprocessing
import pandas as pd
from tqdm import tqdm
import logging

logger = logging.getLogger("qa_pipeline.parallel")


def _process_model_worker(hf_name: str, df_rows: List[Dict[str, str]], batch_size: int, use_cuda: bool):
    """Worker function executado em processo separado.

    Recebe `hf_name` (modelo HF), lista de inputs (cada um com question/context)
    e retorna lista de resultados.
    """
    try:
        from transformers import pipeline
        import torch
    except Exception as e:
        return {"error": str(e)}

    device = 0 if (use_cuda and torch.cuda.is_available()) else -1
    pipe = pipeline("question-answering", model=hf_name, tokenizer=hf_name, device=device)

    results = []
    total = len(df_rows)
    for i in range(0, total, batch_size):
        batch = df_rows[i : i + batch_size]
        inputs = [{"question": r["question"], "context": r["context"]} for r in batch]
        try:
            res = pipe(inputs)
            # normalize single result
            if isinstance(res, dict):
                res = [res]
        except Exception as e:
            res = [{"answer": "", "score": 0.0, "error": str(e)} for _ in inputs]

        # attach shard and original index
        for orig, out in zip(batch, res):
            results.append({**orig, **out})

    return results


class ParallelProcessor:
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or max(1, (multiprocessing.cpu_count() - 1))

    def process_models_parallel(self, model_descriptors: List[Dict[str, Any]], data: pd.DataFrame,
                                batch_size: int = 32, use_cuda: bool = False) -> Dict[str, Any]:
        """Processa múltiplos modelos em paralelo (cada modelo em um processo).

        model_descriptors: lista de dicts com chave `hf_name` e `key`.
        Retorna dicionário { model_key: [results...] }
        """
        results: Dict[str, Any] = {}

        rows = data.to_dict(orient="records")

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_key = {}
            for md in model_descriptors:
                hf = md.get("hf_name")
                key = md.get("key")
                future = executor.submit(_process_model_worker, hf, rows, batch_size, use_cuda)
                future_to_key[future] = key

            for future in tqdm(as_completed(future_to_key), total=len(future_to_key), desc="Processing models"):
                key = future_to_key[future]
                try:
                    model_results = future.result()
                    results[key] = model_results
                except Exception as e:
                    logger.error(f"Error processing {key}: {e}")
                    results[key] = {"error": str(e)}

        return results
