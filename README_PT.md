# QA Pipeline Modular - Análise de Question Answering

Uma pipeline robusta, modular e paralela para processar e analisar respostas de múltiplos modelos de Question Answering (QA) usando a plataforma Hugging Face.

## 📋 Sumário

- [Características](#características)
- [Arquitetura](#arquitetura)
- [Fluxo de Dados](#fluxo-de-dados)
- [Modelos Disponíveis](#modelos-disponíveis)
- [Métricas](#métricas)
- [Instalação](#instalação)
- [Uso](#uso)
- [Exemplos](#exemplos)
- [Estrutura de Saídas](#estrutura-de-saídas)
- [Configuração](#configuração)

---

## ✨ Características

- **Modular**: Arquitetura baseada em componentes independentes e reutilizáveis
- **Paralelo**: Processamento simultâneo de múltiplos modelos usando `ProcessPoolExecutor`
- **Flexível**: Seleção de shards e modelos via CLI ou arquivo YAML
- **Logging estruturado**: Rastreamento detalhado de execução com timestamps
- **Métricas abrangentes**: Análise de confiança, overlap palavra-contexto, concordância entre modelos
- **Exportação multi-formato**: Resultados em CSV, JSON e Markdown
- **Testes automatizados**: Cobertura de componentes principais
- **Poetry**: Gerenciamento de dependências via Poetry

---

## 🏗️ Arquitetura

### Componentes Principais

```
┌─────────────────────────────────────────────────────────────┐
│                    PipelineController                        │
│                (Orquestrador Principal)                      │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
        ┌───────▼──────┐ ┌────▼────┐ ┌────▼──────────┐
        │ ShardLoader  │ │ Model   │ │ Parallel      │
        │ (Dados)      │ │ Selector│ │ Processor     │
        └──────────────┘ └─────────┘ └───────────────┘
                │             │             │
                │             ▼             │
                │      ┌──────────────┐    │
                │      │ Model        │    │
                │      │ Registry     │    │
                │      └──────────────┘    │
                │                         │
                └─────────────┬───────────┘
                              │
                    ┌─────────▼─────────┐
                    │ HF Pipeline       │
                    │ (Modelos: BERT,   │
                    │  DistilBERT,      │
                    │  RoBERTa)         │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼──────────┐
                    │ MetricsCalculator  │
                    │ (Análise e Saídas) │
                    └────────────────────┘
```

### Componentes Detalhados

#### 1. **ShardLoader** (`src/data_loader.py`)
- Descobre e carrega shards CSV do diretório `data/shards/`
- Suporta seleção por padrão glob, lista ou "all"
- Adiciona coluna `_shard` para rastreamento de origem
- Suporta mapeamento de colunas alternativas (`query`→`question`, `text`→`context`)

#### 2. **ModelSelector** (`src/model_selector.py`)
- Registro centralizado de modelos QA disponíveis
- Descritores dinâmicos contendo `key`, `class`, `hf_name`
- Seleção por nome ou "all"

#### 3. **ParallelProcessor** (`src/parallel_processor.py`)
- Executa modelos em processos paralelos separados
- Cada worker instancia um pipeline HF localmente
- Processamento em batches para eficiência
- Suporta CUDA quando disponível

#### 4. **PipelineController** (`src/pipeline_controller.py`)
- Orquestração centralizada do fluxo
- Carregamento de dados → Seleção de modelos → Processamento paralelo → Agregação → Métricas
- Mapeamento automático de esquemas de entrada
- Salvamento de resultados consolidados

#### 5. **MetricsCalculator** (`src/metrics_calculator.py`)
- Cálculo de métricas gerais e por-modelo
- Anotação de overlap palavra-contexto
- Categorização de confiança
- Geração de relatórios (JSON, Markdown, CSV)

#### 6. **BaseQAModel** (`src/base_model.py`)
- Classe abstrata para wrappers de modelos
- Define interface: `load_model()`, `predict()`, `get_metadata()`
- Implementações concretas:
  - `DistilBERTModel`: modelo leve baseado em BERT
  - `RobertaModel`: modelo mais robusto
  - `BERTModel`: BERT completo (Option A - large version)

---

## 🔄 Fluxo de Dados

### Fluxo de Execução

```
1. Leitura de Entrada
   │
   ├─ CLI: args (--shards, --models, ...)
   ├─ YAML Config (opcional): pipeline_config.yaml
   └─ Variáveis de Ambiente: HF_TOKEN, etc.
   │
   ▼
2. Carregamento de Dados (ShardLoader)
   │
   ├─ Descobre shards em data/shards/*.csv
   ├─ Seleciona conforme critério (padrão/lista/all)
   ├─ Concatena em DataFrame único
   └─ Mapeia colunas (query→question, text→context)
   │
   ▼
3. Seleção de Modelos (ModelSelector)
   │
   ├─ Obtém lista de descritores do registry
   ├─ Filtra conforme seleção
   └─ Retorna {key, class, hf_name} por modelo
   │
   ▼
4. Processamento Paralelo (ParallelProcessor)
   │
   ├─ Cria ProcessPoolExecutor (N workers)
   ├─ Cada worker:
   │  ├─ Recebe (hf_name, df_rows, batch_size, use_cuda)
   │  ├─ Instancia pipeline HF localmente
   │  ├─ Processa em batches
   │  └─ Retorna [{"question": ..., "context": ..., "answer": ..., "score": ...}]
   └─ Aguarda conclusão de todos workers
   │
   ▼
5. Agregação de Resultados
   │
   ├─ Combina outputs de todos modelos
   ├─ Adiciona coluna "model" = key do modelo
   └─ DataFrame consolidado: (question, context, answer, score, model, _shard)
   │
   ▼
6. Anotação de Métricas (MetricsCalculator.annotate_overlap)
   │
   ├─ Para cada linha:
   │  ├─ Extrai palavras da resposta
   │  ├─ Verifica presença no contexto
   │  ├─ Calcula overlap_count e overlap_fraction
   └─ Adiciona colunas ao DataFrame
   │
   ▼
7. Cálculo de Métricas Agregadas
   │
   ├─ Overall: mean(score), mean(overlap), total predictions
   ├─ Per-Model: métricas por modelo
   ├─ Comparativa: concordância, distribuição de respostas
   └─ Categórica: distribuição de confiança
   │
   ▼
8. Geração de Saídas
   │
   ├─ results_consolidated.csv: tabela com todas predições + overlap
   ├─ metrics.json: métricas estruturadas
   ├─ metrics_summary.md: relatório legível
   ├─ per_model_metrics.csv: resumo por modelo
   └─ Logs: logs/qa_pipeline_TIMESTAMP.log
   │
   ▼
9. Retorno
   └─ {"results_df": DataFrame, "metrics": dict, "out_dir": Path}
```

### Exemplo de Transformação de Dados

**Entrada (data/shards/shard_001.csv):**
```csv
query,text
What is Python?,Python is a programming language
Who invented Python?,Guido van Rossum created Python
```

**Após ShardLoader:**
```csv
question,context,_shard
What is Python?,Python is a programming language,shard_001.csv
Who invented Python?,Guido van Rossum created Python,shard_001.csv
```

**Após ParallelProcessor (ex: DistilBERT):**
```csv
question,context,answer,score,model,_shard
What is Python?,Python is a programming language,Python,0.95,distilbert,shard_001.csv
Who invented Python?,Guido van Rossum created Python,Guido van Rossum,0.92,distilbert,shard_001.csv
```

**Após MetricsCalculator.annotate_overlap:**
```csv
question,context,answer,score,model,_shard,overlap_count,overlap_fraction
What is Python?,Python is a programming language,Python,0.95,distilbert,shard_001.csv,1,1.0
Who invented Python?,Guido van Rossum created Python,Guido van Rossum,0.92,distilbert,shard_001.csv,2,1.0
```

---

## 🤖 Modelos Disponíveis

| Modelo | Checkpoint HF | Tamanho | Descrição |
|--------|---------------|--------|-----------|
| **distilbert** | `distilbert-base-cased-distilled-squad` | 268MB | Versão destilada, rápida e leve |
| **roberta** | `deepset/roberta-base-squad2` | ~498MB | RoBERTa fine-tuned em SQuAD 2.0 |
| **bert** | `bert-large-uncased-whole-word-masking-finetuned-squad` | ~1.3GB | BERT completo, mais preciso |

**Seleção via CLI:**
```bash
# Um modelo
poetry run python -m src.main --models distilbert

# Múltiplos
poetry run python -m src.main --models distilbert roberta

# Todos
poetry run python -m src.main --models all
```

---

## 📊 Métricas

### Métricas por Predição

Cada linha do `results_consolidated.csv` inclui:

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `question` | str | Pergunta de entrada |
| `context` | str | Contexto/passagem |
| `answer` | str | Resposta gerada |
| `score` | float | Confiança do modelo [0.0, 1.0] |
| `model` | str | Nome do modelo (`distilbert`, `roberta`, `bert`) |
| `_shard` | str | Arquivo de origem |
| `overlap_count` | int | **Palavras da resposta presentes no contexto** |
| `overlap_fraction` | float | **overlap_count / total palavras na resposta** |

### Métricas Agregadas (metrics.json)

#### Overall
```json
{
  "overall": {
    "total_predictions": 300,
    "mean_score": 0.87,
    "median_score": 0.91,
    "avg_overlap_fraction": 0.64,
    "avg_overlap_count": 3.2
  }
}
```

**Descrição:**
- `total_predictions`: total de predições (shards × modelos)
- `mean_score`: confiança média
- `avg_overlap_fraction`: fração média de palavras da resposta no contexto
- `avg_overlap_count`: número médio de palavras coincidentes

#### Per-Model
```json
{
  "per_model": {
    "distilbert": {
      "count": 100,
      "mean_score": 0.85,
      "median_score": 0.90,
      "avg_overlap_fraction": 0.62,
      "avg_overlap_count": 3.1
    },
    "bert": {
      "count": 100,
      "mean_score": 0.92,
      "median_score": 0.94,
      "avg_overlap_fraction": 0.68,
      "avg_overlap_count": 3.4
    }
  }
}
```

#### Comparativa
```json
{
  "comparative": {
    "avg_unique_answers": 2.1
  }
}
```

**Descrição:** número médio de respostas únicas por (question, context) — indica concordância entre modelos.

#### Categórica
```json
{
  "categorical": {
    "low_risk": 234,
    "medium_risk": 45,
    "high_risk": 21
  }
}
```

**Categorização por Confiança:**
- `low_risk`: score ≥ 0.8
- `medium_risk`: 0.5 ≤ score < 0.8
- `high_risk`: score < 0.5

### Interpretação da Métrica de Overlap

**Overlap Palavra-Contexto:**

Mede o grau em que a resposta está "ancorada" no contexto fornecido.

**Exemplos:**
```
Contexto: "Paris é a capital da França, conhecida por monumentos históricos."
Resposta: "Paris"
→ overlap_count=1, overlap_fraction=1.0 (100% das palavras da resposta estão no contexto)

Contexto: "O gato dorme na cama."
Resposta: "animal dormindo"
→ overlap_count=1 (apenas "dormindo" está no contexto, "animal" não)
→ overlap_fraction=0.5 (50% das palavras estão presentes)

Contexto: "Python é uma linguagem."
Resposta: "JavaScript é melhor"
→ overlap_count=0, overlap_fraction=0.0 (nenhuma palavra matches)
```

**Interpretação:**
- `overlap_fraction ≈ 1.0`: Resposta altamente suportada pelo contexto (boa)
- `overlap_fraction ≈ 0.5`: Resposta parcialmente suportada (moderado)
- `overlap_fraction ≈ 0.0`: Resposta pouco ancorada no contexto (alerta)

---

## 🚀 Instalação

### Pré-requisitos
- Python ≥ 3.8.1
- Poetry ≥ 1.2

### Passos

1. **Clone o repositório:**
```bash
git clone <seu-repo> dashboard_pln
cd dashboard_pln
```

2. **Instale as dependências via Poetry:**
```bash
poetry install
```

3. (Opcional) **Configure HuggingFace Token** para acesso a modelos privados:
```bash
# Criar arquivo .env
echo "HF_TOKEN=seu_token_aqui" > .env
```

4. **Verifique a instalação:**
```bash
poetry run pytest -q
```

---

## 📝 Uso

### Linha de Comando (CLI)

```bash
poetry run python -m src.main [opções]
```

**Opções:**

| Opção | Padrão | Descrição |
|-------|--------|-----------|
| `--shards` | `["all"]` | Shards a processar: `all`, glob (ex: `shard_0*`), ou lista |
| `--models` | `["all"]` | Modelos a usar: `distilbert`, `roberta`, `bert`, ou `all` |
| `--batch-size` | `32` | Tamanho do lote para inferência |
| `--workers` | `auto` | Número de processos paralelos |
| `--max-samples` | `None` | Limita samples para teste (ex: `200`) |
| `--output-dir` | `outputs` | Diretório de saída |
| `--log-dir` | `logs` | Diretório de logs |
| `--config` | `None` | Arquivo YAML de configuração (opcional) |

---

## 💡 Exemplos

### Exemplo 1: Rodar um único shard com todos modelos

```bash
poetry run python -m src.main --shards shard_055.csv --models all
```

Saída:
```
2026-02-02 12:30:15 | INFO | qa_pipeline | Starting pipeline run
2026-02-02 12:30:16 | INFO | qa_pipeline | Mapping input columns: 'query'->'question', 'text'->'context'
2026-02-02 12:30:16 | INFO | qa_pipeline | CUDA available: False
2026-02-02 12:30:45 | INFO | qa_pipeline | Saved consolidated results to outputs/20260202_123045/results_consolidated.csv
2026-02-02 12:30:46 | INFO | qa_pipeline | Report saved: outputs/20260202_123045/metrics_summary.md
```

### Exemplo 2: Rodar com seleção de shards e modelo específico

```bash
poetry run python -m src.main --shards shard_001.csv shard_002.csv --models bert --max-samples 50
```

### Exemplo 3: Rodar via arquivo YAML

**config/pipeline_config.yaml:**
```yaml
shards:
  - "shard_0*.csv"
models:
  - "distilbert"
  - "roberta"
batch_size: 16
workers: 2
max_samples: 100
output_dir: "outputs_custom"
log_dir: "logs_custom"
```

```bash
poetry run python -m src.main --config config/pipeline_config.yaml
```

### Exemplo 4: Teste rápido com dados limitados

```bash
poetry run python -m src.main --shards shard_055.csv --models distilbert --max-samples 10 --output-dir outputs_test
```

---

## 📂 Estrutura de Saídas

### Diretório de Execução

```
outputs/
└── 20260202_123045/              # Timestamp: YYYYMMDD_HHMMSS
    ├── results_consolidated.csv   # Tabela completa (predições + métricas)
    ├── metrics.json               # Métricas estruturadas
    ├── metrics_summary.md         # Relatório legível
    └── per_model_metrics.csv      # Resumo por modelo
```

### results_consolidated.csv

Tabela com todas predições e colunas de overlap:

```csv
question,context,answer,score,model,_shard,overlap_count,overlap_fraction
"What is Python?","Python is a...",Python,0.95,distilbert,shard_001.csv,1,1.0
"What is Python?","Python is a...",Programming language,0.89,roberta,shard_001.csv,2,1.0
```

**Uso:** Análise manual, exportação para BI, validação detalhada

### metrics_summary.md

Relatório formatado legível para compartilhamento:

```markdown
# Metrics Summary

## Overall
- total_predictions: 300
- mean_score: 0.87
- avg_overlap_fraction: 0.64

## Per Model
### distilbert
- count: 100
- mean_score: 0.85
- avg_overlap_fraction: 0.62

### bert
- count: 100
- mean_score: 0.92
- avg_overlap_fraction: 0.68
```

### per_model_metrics.csv

Resumo por modelo para comparação rápida:

```csv
model,count,mean_score,median_score,avg_overlap_fraction,avg_overlap_count
distilbert,100,0.85,0.90,0.62,3.1
roberta,100,0.88,0.92,0.65,3.2
bert,100,0.92,0.94,0.68,3.4
```

---

## ⚙️ Configuração

### Arquivo YAML (config/pipeline_config.yaml)

```yaml
# Shards para processar
shards:
  - "all"  # ou ["shard_001.csv", "shard_002.csv"]

# Modelos para executar
models:
  - "all"  # ou ["distilbert", "bert"]

# Inferência
batch_size: 32
workers: null  # Auto-detect CPU cores

# Limitações (para teste)
max_samples: null  # null = sem limite

# Diretórios
output_dir: "outputs"
log_dir: "logs"
```

### Variáveis de Ambiente

```bash
# .env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
CUDA_VISIBLE_DEVICES=0  # Especifique GPU se disponível
PYTHONPATH=.
```

---

## 🧪 Testes

### Rodar todos os testes:

```bash
poetry run pytest -q
```

### Rodar teste específico:

```bash
poetry run pytest tests/test_model_selector.py -v
```

### Rodar apenas testes de overlap:

```bash
poetry run pytest tests/tests_metrics_overlap.py -v
```

---

## 📦 Dependências Principais

| Pacote | Versão | Uso |
|--------|--------|-----|
| pandas | ≥1.3 | Manipulação de dados |
| transformers | ≥4.20 | Modelos HF QA |
| torch | ≥1.10 | Backend de ML |
| pyyaml | ≥5.4 | Configuração |
| tqdm | ≥4.60 | Barras de progresso |
| huggingface-hub | ≥0.12 | Autenticação HF |

---

## 📋 Estrutura de Projeto

```
dashboard_pln/
├── src/
│   ├── __init__.py
│   ├── base_model.py              # Classe abstrata
│   ├── data_loader.py             # Carregador de shards
│   ├── logger_config.py           # Logging
│   ├── main.py                    # Entrada CLI
│   ├── metrics_calculator.py      # Cálculo de métricas
│   ├── model_selector.py          # Registro de modelos
│   ├── parallel_processor.py      # Processamento paralelo
│   ├── pipeline_controller.py     # Orquestrador
│   └── models/
│       ├── __init__.py
│       ├── distilbert_model.py    # Wrapper DistilBERT
│       ├── roberta_model.py       # Wrapper RoBERTa
│       └── bert_model.py          # Wrapper BERT
├── tests/
│   ├── test_data_loader.py
│   ├── test_model_selector.py
│   └── tests_metrics_overlap.py
├── data/
│   └── shards/                    # Arquivos CSV de entrada
│       ├── shard_000.csv
│       ├── shard_001.csv
│       └── ...
├── config/
│   └── pipeline_config.yaml       # Configuração YAML
├── logs/                          # Saídas de log
├── outputs/                       # Resultados
├── .env                           # Variáveis de ambiente
├── pyproject.toml                 # Dependências Poetry
├── README_PT.md                   # Este arquivo
└── projeto_av02_pln_lucas_cavalcante.ipynb  # Notebook de análise
```

---

## 🐛 Troubleshooting

### Erro: "ModuleNotFoundError: No module named 'src'"

**Solução:** Execute pelo Poetry:
```bash
poetry run python -m src.main ...
```

### Erro: "CUDA out of memory"

**Solução:** Reduza batch size ou mude para CPU:
```bash
poetry run python -m src.main --batch-size 8
```

### Modelos não são baixados

**Solução:** Verifique token HF:
```bash
poetry run huggingface-cli login
# ou
export HF_TOKEN=seu_token
```

### Logs muito grandes

**Solução:** Limpe diretório `logs/`:
```bash
rm logs/qa_pipeline_*.log
```

---

## 📞 Contato & Suporte

Para dúvidas ou issues, abra uma issue no repositório ou entre em contato com a equipe de desenvolvimento.

---

## 📝 Licença

Este projeto está disponível sob a licença MIT. Veja `LICENSE` para detalhes.

---

**Última atualização:** Fevereiro 2, 2026
**Versão da Pipeline:** 2.0 (com overlap palavra-contexto)
