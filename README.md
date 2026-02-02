# QA Pipeline - Question Answering Modular & Paralelo

**[Versão em Português →](README_PT.md)**

Uma pipeline robusta, modular e paralela para processar e analisar respostas de múltiplos modelos de Question Answering (QA) usando a plataforma Hugging Face.

## Tabela de Conteúdos

- [Características](#-characteristics)
- [Arquitetura](#-architecture)
- [Fluxo de Dados](#-data-flow)
- [Modelos](#-models)
- [Métricas](#-metrics)
- [Instalação](#-setup)
- [Uso](#-usage)
- [Exemplos](#-examples)
- [Saídas](#-outputs)

---

## ✨ Características

### 🔄 Pipeline Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   INPUT DATA    │───▶│ DATA LOADING    │───▶│ MODEL SELECTION │
│                 │    │                 │    │                 │
│ • CSV Shards   │    │ • Discovery     │    │ • Registry      │
│ • CLI Args      │    │ • Validation    │    │ • Descriptors   │
│ • YAML Config   │    │ • Mapping       │    │ • Device Alloc  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│PARALLEL PROCESSING│───▶│RESULTS AGGREGATION│───▶│METRICS CALCULATION│
│                 │    │                 │    │                 │
│ • Multi-Process │    │ • Collection    │    │ • Overlap       │
│ • Batch Size    │    │ • Unification   │    │ • Performance   │
│ • HF Pipelines  │    │ • Annotation    │    │ • Consensus     │
│ • Error Handle  │    │ • Traceability   │    │ • Risk Analysis │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                           ┌─────────────────┐
                                           │ OUTPUT STORAGE  │
                                           │                 │
                                           │ • Timestamp Dir  │
                                           │ • CSV Files     │
                                           │ • JSON Data     │
                                           │ • MD Reports    │
                                           └─────────────────┘
```

### 📋 Stage 1: Input & Configuration
**Entry Point**: `src/main.py` (CLI Interface)
- **Configuration Sources**:
  - Command-line arguments (`--shards`, `--models`, `--batch-size`)
  - YAML configuration (`config/pipeline_config.yaml`)
  - Environment variables
- **Data Sources**: CSV shards in `data/shards/` directory
- **Validation**: Schema validation and format checking

### 📂 Stage 2: Data Loading (`src/data_loader.py`)
**Flexible Data Ingestion**:
- **Discovery**: Glob patterns for CSV file detection
- **Schema Mapping**: Auto-detect column patterns
  - `question`/`context` ↔ `query`/`text`
- **Processing**: Concatenation with shard traceability
- **Output**: Unified DataFrame with `_shard` column

### 🤖 Stage 3: Model Selection (`src/model_selector.py`)
**Dynamic Model Registry**:
- **Available Models**:
  - `distilbert`: `distilbert-base-cased-distilled-squad`
  - `roberta`: `deepset/roberta-base-squad2`  
  - `bert`: `bert-large-uncased-whole-word-masking-finetuned-squad`
- **Descriptors**: `{key, hf_name, device}` metadata
- **Device Allocation**: Automatic CUDA/CPU detection

### ⚡ Stage 4: Parallel Processing (`src/parallel_processor.py`)
**High-Performance Execution**:
- **Architecture**: ProcessPoolExecutor (true parallelism)
- **Isolation**: Each model in separate process (no GIL conflicts)
- **Batch Processing**: Configurable batch sizes
- **Integration**: Hugging Face `pipeline("question-answering")`
- **Error Recovery**: Fallback responses for failures
- **Output Format**: `{answer, score, start, end}` per prediction

### 🔄 Stage 5: Results Aggregation (`src/pipeline_controller.py`)
**Data Unification**:
- **Collection**: Gather results from all model processes
- **Enrichment**: Add `model` and processing metadata
- **Consolidation**: Create unified DataFrame
- **Overlap Analysis**: Model comparison annotations
- **Traceability**: Shard and model lineage tracking

### 📊 Stage 6: Metrics Calculation (`src/metrics_calculator.py`)
**Comprehensive Analytics**:

**Overlap Analysis**:
- `overlap_count`: Number of identical answers per question
- `overlap_fraction`: Consensus ratio across models

**Performance Metrics**:
- Score distributions (mean, median, std, percentiles)
- Confidence intervals and error analysis
- Model-specific performance statistics

**Comparative Analysis**:
- Cross-model consensus evaluation
- Performance ranking and comparison
- Answer similarity analysis

**Risk Categorization**:
- Low/Medium/High confidence based on scores
- Uncertainty quantification
- Decision support metrics

### 💾 Stage 7: Output Storage (`outputs/YYYYMMDD_HHMMSS/`)
**Structured Results**:
- **Primary Data**:
  - `results_consolidated.csv`: All predictions with annotations
  - `per_model_metrics.csv`: Flattened model statistics
  
- **Analytics**:
  - `metrics.json`: Complete metrics data structure
  - `metrics_summary.md`: Human-readable analysis report
  
- **Traceability**:
  - Timestamp directory organization
  - Model configuration export
  - Processing logs and error tracking

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

```bash
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
```