# QA Analysis Dashboard

A comprehensive Streamlit dashboard for Question Answering (QA) model analysis and comparison with parallel processing, disk-based caching, and full import/export support.

## 🚀 Features

### Core Functionality
- **🤖 Dual Model Support**: DistilBERT (fast) and RoBERTa (accurate) QA models
- **⚡ Parallel Processing**: Simultaneous model execution with real-time progress tracking
- **💾 Disk-Based Caching**: Persistent caching for datasets and results
- **📁 Hybrid Data Interface**: Upload files or browse existing datasets
- **📊 Comprehensive Visualization**: Interactive charts and detailed analysis
- **📦 Multi-Sheet Export**: Complete analysis packages with multiple CSV sheets
- **📝 Error Logging**: Detailed pipeline logs for debugging

### Advanced Features
- **Real-time Progress Tracking**: Live progress bars and status updates
- **Model Comparison**: Head-to-head performance analysis
- **Statistical Analysis**: Score distributions, overlap analysis, correlation studies
- **Export Management**: Browse and import existing analysis files
- **Quality Validation**: Automatic dataset validation and quality reports
- **Error Recovery**: Robust error handling with detailed logging

## 🛠️ Installation

### Using Poetry (Recommended)

```bash
# Clone or download the project
cd qa-analysis-dashboard

# Install dependencies with Poetry
poetry install

# Activate virtual environment
poetry shell

# Start the dashboard
poetry run streamlit run streamlit_app.py
```

### Manual Installation

```bash
# Install dependencies manually
pip install streamlit pandas numpy torch transformers plotly matplotlib seaborn python-dotenv scikit-learn tqdm

# Start the dashboard
streamlit run streamlit_app.py
```

## 📋 Project Structure

```
qa-analysis-dashboard/
├── streamlit_app.py          # Main dashboard entry point
├── pages/                   # Streamlit page modules
│   ├── dashboard.py          # Main overview page
│   ├── data_management.py     # Data upload and browsing
│   ├── model_analysis.py      # Model configuration and processing
│   └── results_visualization.py # Results and export browser
├── utils/                    # Utility modules
│   ├── data_manager.py       # Unified data management with caching
│   ├── parallel_processor.py  # Parallel model processing
│   ├── import_export.py      # Import/export system
│   ├── helpers.py           # Helper functions
│   └── metrics.py           # Analysis metrics
├── config/                   # Configuration
│   └── settings.py           # Project settings
├── models/                   # Model wrappers
│   ├── __init__.py
│   ├── base_model.py
│   ├── distilbert_model.py
│   └── roberta_model.py
├── data/                     # Data handling
│   ├── __init__.py
│   └── dataloader.py
├── output/                   # Generated outputs
├── cache/                    # Disk-based cache
└── logs/                     # Error and processing logs
```

## 📖 Usage Guide

### 1. Data Loading
- **Upload Files**: Use the "Upload Data" tab to upload CSV files
- **Browse Datasets**: Use "Browse Datasets" tab to select existing shards
- **Data Validation**: Automatic quality checks and validation reports

### 2. Model Configuration
- **Model Selection**: Choose DistilBERT, RoBERTa, or both
- **Processing Parameters**: Configure batch size, confidence thresholds
- **Hardware Detection**: Automatic GPU/CPU detection and optimization

### 3. Analysis Execution
- **Parallel Processing**: Models run simultaneously for efficiency
- **Progress Tracking**: Real-time progress bars and status updates
- **Error Handling**: Comprehensive error logging and recovery

### 4. Results Visualization
- **Summary Dashboard**: Overview metrics and performance stats
- **Model Comparison**: Side-by-side model analysis
- **Detailed Results**: Interactive tables with filtering
- **Export Browser**: Browse existing analysis files

### 5. Export Management
- **Multi-Sheet CSV**: Comprehensive analysis packages
- **Visualization Charts**: High-resolution plots and charts
- **Statistical Reports**: Detailed analysis summaries
- **Error Logs**: Complete processing logs

## 🔧 Configuration

### Environment Variables
```bash
# Hugging Face Token (optional)
HF_TOKEN=your_hf_token_here

# Custom Model Paths (optional)
DISTILBERT_MODEL=path/to/distilbert
ROBERTA_MODEL=path/to/roberta
```

### Settings Configuration
The dashboard uses a hierarchical configuration system:

- **`config/settings.py`**: Main configuration file
- **Environment Variables**: Override with environment
- **Runtime Detection**: Automatic hardware detection
- **Caching Strategy**: Disk-based caching for persistence

## 📊 Data Format

### Input Dataset Format
```csv
_id,question,context,title
1,"What is the capital of France?","France is a country in Western Europe. Its capital is Paris...","France"
2,"Who wrote Romeo and Juliet?","Romeo and Juliet is a tragedy written by William Shakespeare...","Literature"
```

### Output Analysis Format
The dashboard generates comprehensive analysis packages with multiple sheets:

- **`resultados_completos_*.csv`**: Full results with both model outputs
- **`distilbert_top10_melhores_*.csv`**: Top 10 DistilBERT results
- **`roberta_top10_melhores_*.csv`**: Top 10 RoBERTa results
- **`global_top10_melhores_*.csv`**: Overall best 10 results
- **`discordancias_*.csv`**: Cases where models disagree significantly
- **`resumo_estatistico_*.csv`**: Statistical summary
- **`metadata_*.json`**: Complete analysis metadata

## 🎯 Performance Metrics

### Model Performance Indicators
- **Confidence Scores**: Model prediction confidence (0.0-1.0)
- **Overlap Analysis**: Word overlap between context and answer
- **Processing Speed**: Tokens/second and examples/minute
- **Memory Usage**: GPU/CPU memory consumption
- **Error Rates**: Failed processing and error types

### Statistical Analyses
- **Score Distributions**: Histograms and density plots
- **Model Correlation**: Scatter plots comparing models
- **Performance Comparison**: Box plots and violin plots
- **Temporal Analysis**: Processing time over dataset size
- **Quality Metrics**: Data quality and validation scores

## 🔍 Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Verify model access
python -c "from transformers import pipeline; print('Models accessible')"
```

#### Memory Issues
```bash
# Reduce batch size in model configuration
# Use smaller datasets for testing
# Monitor memory usage with task manager
```

#### Import/Export Problems
```bash
# Check file permissions
ls -la output/
# Verify CSV format
python -c "import pandas as pd; print(pd.read_csv('your_file.csv').head())"
```

### Debug Mode
Enable detailed logging:
```python
# Set logging level
import logging
logging.basicConfig(level=logging.DEBUG)

# Check error logs
tail -f logs/qa_pipeline_*.log
```

## 🚀 Deployment

### Local Development
```bash
# Development mode with hot reload
poetry run streamlit run streamlit_app.py

# With custom port
poetry run streamlit run streamlit_app.py --server.port 8080
```

### Production Deployment
```bash
# With authentication
streamlit run streamlit_app.py --server.headless true

# Behind proxy
streamlit run streamlit_app.py --server.enableCORS false
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .
RUN poetry install --no-dev

EXPOSE 8501
CMD ["poetry", "run", "streamlit", "run", "streamlit_app.py", "--server.headless", "true"]
```

## 🧪 Testing

### Run Test Suite
```bash
# Basic functionality test
poetry run python simple_test.py

# Full pipeline test
poetry run python test_pipeline.py
```

### Test Coverage
- ✅ Import validation
- ✅ Configuration testing
- ✅ Data manager functionality
- ✅ Parallel processing
- ✅ Import/export operations
- ✅ Error handling
- ✅ Cache management

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd qa-analysis-dashboard

# Setup development environment
poetry install
poetry shell

# Run tests
python simple_test.py
```

### Code Style
```bash
# Format code
poetry run black .

# Sort imports
poetry run isort .

# Type checking
poetry run mypy .
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Support

For issues, questions, or contributions:
- 📧 Create an issue in the project repository
- 📧 Check the troubleshooting section
- 📖 Review the comprehensive documentation
- 🧪 Run the test suite for validation

## 🔄 Changelog

### Version 0.1.0
- ✅ Initial Streamlit dashboard implementation
- ✅ Parallel model processing system
- ✅ Disk-based caching with persistence
- ✅ Comprehensive import/export functionality
- ✅ Real-time progress tracking
- ✅ Error logging and recovery system
- ✅ Multi-sheet analysis packages
- ✅ Interactive visualization dashboards
- ✅ Poetry-based dependency management