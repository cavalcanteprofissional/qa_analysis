# QA Pipeline

Implementação modular de pipeline de Question Answering com execução paralela.

## Data Flow

The pipeline follows a structured flow from input data to comprehensive analysis:

```
Input Data → Data Loading → Model Selection → Parallel Processing → Results Aggregation → Metrics Calculation → Output Storage
```

### 1. Input Stage
- **Entry Point**: `src/main.py` - CLI interface
- **Configuration**: Command-line args or `config/pipeline_config.yaml`
- **Data Source**: CSV shards in `data/shards/` directory

### 2. Data Loading (`src/data_loader.py`)
- Discovers and loads CSV files with flexible selection
- Supports columns: `question`/`context` or `query`/`text`
- Adds traceability with `_shard` column

### 3. Model Selection (`src/model_selector.py`)
- Available models: DistilBERT, RoBERTa, BERT
- Dynamic instantiation via model descriptors
- Automatic CUDA detection and device allocation

### 4. Parallel Processing (`src/parallel_processor.py`)
- ProcessPoolExecutor for true parallelism
- Each model runs in isolated process
- Batch processing with configurable sizes
- Hugging Face pipeline integration

### 5. Results Aggregation (`src/pipeline_controller.py`)
- Collects results from all model processes
- Adds model identification columns
- Creates unified DataFrame with overlap annotations

### 6. Metrics Calculation (`src/metrics_calculator.py`)
- **Overlap Analysis**: Count and fraction of identical answers
- **Performance Metrics**: Score distributions, confidence intervals
- **Comparative Analysis**: Cross-model consensus evaluation
- **Risk Categorization**: Low/medium/high confidence classification

### 7. Output Storage
- **Directory**: `outputs/YYYYMMDD_HHMMSS/`
- **Files**:
  - `results_consolidated.csv`: All predictions with annotations
  - `metrics.json`: Detailed metrics data
  - `metrics_summary.md`: Human-readable report
  - `per_model_metrics.csv`: Individual model statistics

## Usage Examples

```bash
# Run on all shards with all models
python -m src.main --shards all --models all

# Run specific shards and models
python -m src.main --shards shard_001.csv shard_002.csv --models distilbert roberta

# Custom configuration
python -m src.main --shards all --models all --batch-size 16 --workers 4
```

## Installation

See `pyproject.toml` for dependencies. Use Poetry to install:

```bash
poetry install
poetry run python -m src.main --shards all --models all
```
qa-pipeline/
├── src/
│   ├── base_model.py           # Classe base abstrata para modelos
│   ├── distilbert_model.py     # Implementação DistilBERT
│   ├── roberta_model.py        # Implementação RoBERTa
│   ├── data_loader.py          # Carregamento de shards CSV
│   ├── pipeline_controller.py  # Orquestração da pipeline
│   ├── metrics_calculator.py   # Cálculo de métricas
│   ├── result_exporter.py      # Exportação de resultados
│   ├── logger_config.py        # Configuração de logging
│   └── main.py                 # Ponto de entrada
├── config/
│   ├── model_config.yaml       # Configurações dos modelos
│   └── pipeline_config.yaml    # Configurações da pipeline
├── data/
│   └── shards/                 # CSV shards
│       ├── shard_001.csv
│       ├── shard_002.csv
│       └── ...
├── logs/                       # Logs da pipeline
├── outputs/                    # Resultados e métricas
├── tests/
├── pyproject.toml
└── README.md


outputs/
└── run_20240115_143022/          # Timestamp da execução
    ├── logs/
    │   └── pipeline_20240115_143022.log
    ├── results/
    │   ├── aggregated_results.csv
    │   ├── per_shard/
    │   │   ├── shard_001_results.csv
    │   │   └── ...
    │   └── per_model/
    │       ├── distilbert_results.csv
    │       └── roberta_results.csv
    ├── metrics/
    │   ├── summary_report.md
    │   ├── detailed_metrics.json
    │   ├── visualizations/
    │   │   ├── scores_distribution.png
    │   │   ├── overlap_comparison.png
    │   │   └── ...
    │   └── comparative_analysis.csv
    └── config/
        └── pipeline_config_used.yaml