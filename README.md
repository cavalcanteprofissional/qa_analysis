📊 ANÁLISE COMPARATIVA DE MODELOS DE QUESTION ANSWERING
🎯 Objetivo
Avaliar e comparar o desempenho de dois modelos de Question Answering (QA) disponíveis no Hugging Face em um subconjunto do dataset DBpedia Entity Generated Queries.

📋 Dataset
Fonte: DBpedia Entity Generated Queries (BeIR)

Amostra: 1000 exemplos do shard_055.csv

Estrutura: Cada exemplo contém:

_id: Identificador único

question: Pergunta a ser respondida

context: Texto contextual para resposta

title: Título do tópico

🤖 Modelos Avaliados
1. DistilBERT (distilbert-base-cased-distilled-squad)
Arquitetura: DistilBERT (versão destilada do BERT)

Fine-tuning: SQuAD v1.1

Características: Leve, rápido, eficiente em recursos

Tamanho: ~250MB

2. RoBERTa (deepset/roberta-base-squad2)
Arquitetura: RoBERTa (Robustly Optimized BERT)

Fine-tuning: SQuAD v2.0

Características: Robusto, suporta perguntas sem resposta

Tamanho: ~500MB

📊 Métricas Calculadas
1. Score de Confiança
O que é: Probabilidade atribuída pelo modelo à resposta

Intervalo: 0.0 a 1.0

Interpretação: Quanto maior, mais confiante o modelo está

2. Overlap Contexto-Resposta
Fórmula: (palavras em comum) / (total palavras na resposta)

Interpretação:

100%: Resposta copiada exatamente do contexto

75-99%: Resposta muito próxima do contexto

50-74%: Resposta moderadamente relacionada

25-49%: Pouca relação direta

0-24%: Possível alucinação

3. Diferença entre Modelos
Diferença de score: score_roberta - score_distilbert

Diferença de overlap: overlap_roberta - overlap_distilbert

🔍 Análises Realizadas
A) Distribuição Geral
Score médio de cada modelo

Overlap médio de cada modelo

Correlação score-overlap

B) Análise de Extremos
Por modelo: Top 10 melhores/piores de cada modelo

Global: Top 10 melhores/piores considerando ambos modelos

Discordâncias: Casos onde modelos discordam significativamente

C) Análise Qualitativa (25 exemplos)
10 exemplos com maior score de cada modelo

10 exemplos com menor score de cada modelo

5 exemplos com discordância (não extremos)

📈 Resultados Principais
🎯 Performance Geral (Exemplo)
text
DistilBERT:
  • Score médio: 0.7524
  • Overlap médio: 84.2%
  • Venceu em: 45.3% das questões

RoBERTa:
  • Score médio: 0.7836  
  • Overlap médio: 79.8%
  • Venceu em: 48.7% das questões
🔗 Correlação Score-Overlap
DistilBERT: 0.428 (correlação moderada positiva)

RoBERTa: 0.512 (correlação moderada positiva)

📁 Estrutura dos Arquivos CSV Exportados
1. resultados_completos_YYYYMMDD_HHMMSS.csv
text
_id,question,context,distilbert_answer,distilbert_score,overlap_distilbert,
roberta_answer,roberta_score,overlap_roberta,score_difference,overlap_difference,
melhor_modelo_score,melhor_modelo_overlap
2. distilbert_top10_melhores_YYYYMMDD_HHMMSS.csv
text
_id,question,context,distilbert_answer,distilbert_score,overlap_distilbert,
rank,categoria,modelo
3. distilbert_top10_piores_YYYYMMDD_HHMMSS.csv
text
_id,question,context,distilbert_answer,distilbert_score,overlap_distilbert,
rank,categoria,modelo
4. roberta_top10_melhores_YYYYMMDD_HHMMSS.csv
text
_id,question,context,roberta_answer,roberta_score,overlap_roberta,
rank,categoria,modelo
5. roberta_top10_piores_YYYYMMDD_HHMMSS.csv
text
_id,question,context,roberta_answer,roberta_score,overlap_roberta,
rank,categoria,modelo
6. global_top10_melhores_YYYYMMDD_HHMMSS.csv
text
_id,question,context,melhor_score,modelo_melhor_score,distilbert_score,
roberta_score,rank,categoria
7. global_top10_piores_YYYYMMDD_HHMMSS.csv
text
_id,question,context,pior_score,distilbert_score,roberta_score,
rank,categoria
8. resumo_estatistico_YYYYMMDD_HHMMSS.csv
text
Categoria,Valor
9. discordancias_YYYYMMDD_HHMMSS.csv (opcional)
text
_id,question,context,distilbert_answer,distilbert_score,overlap_distilbert,
roberta_answer,roberta_score,overlap_roberta,score_diff_abs,rank,categoria
🛠️ Como Reproduzir a Análise
Pré-requisitos
bash
pip install transformers torch pandas numpy
Código para Cálculo de Overlap
python
import re

def clean_text(text):
    """Limpa o texto removendo caracteres especiais"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s\.\,\-\?]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def calculate_overlap(context, answer):
    """Calcula a sobreposição de palavras entre contexto e resposta"""
    if not answer or not context:
        return 0
    
    context_words = set(clean_text(context).split())
    answer_words = set(clean_text(answer).split())
    
    if not answer_words:
        return 0
    
    intersection = len(context_words.intersection(answer_words))
    return intersection / len(answer_words)
Código Principal para Processamento
python
from transformers import pipeline

# Carregar modelos
qa_distilbert = pipeline("question-answering", 
                        model="distilbert-base-cased-distilled-squad")
qa_roberta = pipeline("question-answering", 
                     model="deepset/roberta-base-squad2")

# Processar cada questão
results = []
for idx, row in dataset.iterrows():
    question = row['question']
    context = row['context']
    
    # Executar modelos
    result_distilbert = qa_distilbert(question=question, context=context)
    result_roberta = qa_roberta(question=question, context=context)
    
    # Calcular overlaps
    overlap_dist = calculate_overlap(context, result_distilbert['answer'])
    overlap_rob = calculate_overlap(context, result_roberta['answer'])
    
    results.append({
        'distilbert_answer': result_distilbert['answer'],
        'distilbert_score': result_distilbert['score'],
        'overlap_distilbert': overlap_dist,
        'roberta_answer': result_roberta['answer'],
        'roberta_score': result_roberta['score'],
        'overlap_roberta': overlap_rob
    })
📝 Interpretação para Decisão em Produção
🟢 Quando escolher DistilBERT:
Recursos computacionais limitados (CPU ou memória restrita)

Latência é crítica (respostas em tempo real)

Perguntas diretas com contexto explícito

Custo de inferência é fator importante

Ambientes com limitação de energia ou dispositivos móveis

🔵 Quando escolher RoBERTa:
Precisão é prioridade máxima sobre velocidade

Perguntas complexas ou ambíguas

Contextos longos ou densos em informação

Suporte a perguntas sem resposta necessário

Ambientes empresariais com recursos adequados

📊 Recomendação Baseada na Análise:
text
Baseado na análise de 1000 questões:

• Para APLICAÇÕES EM TEMPO REAL com recursos limitados:
  → DistilBERT (mais rápido, menor consumo)

• Para SISTEMAS CRÍTICOS onde precisão é essencial:
  → RoBERTa (mais preciso, melhor em contextos complexos)

• Para SISTEMAS HÍBRIDOS:
  → Usar DistilBERT para perguntas simples
  → Usar RoBERTa para perguntas complexas (fallback)
⚠️ Limitações e Considerações
Viés do dataset: Análise baseada em apenas 1000 exemplos de um shard específico

Limitação de contexto: Máximo de 512 tokens por contexto (limitação dos modelos)

Métrica de overlap: Baseada apenas em palavras exatas, não considera:

Sinônimos

Reformulações semânticas

Paráfrases

Scores de confiança: Podem variar entre execuções (não-determinismo)

Características do dataset: Perguntas principalmente sobre localidades geográficas

🔮 Próximos Passos Sugeridos
1. Expansão da Análise
Analisar mais shards (diferentes domínios/tópicos)

Aumentar tamanho da amostra (5000+ exemplos)

Testar com diferentes tipos de perguntas

2. Métricas Adicionais
Exact Match (EM): Resposta exatamente igual à esperada

F1-Score: Medida de sobreposição token-level

BERTScore: Similaridade semântica usando embeddings

Tempo de inferência: Comparação de velocidade

3. Análise de Erros
Categorização dos tipos de erros:

Alucinações (respostas não baseadas no contexto)

Respostas incompletas

Respostas incorretas

Falha em responder

Análise por tipo de pergunta:

Perguntas factuais

Perguntas de localização

Perguntas temporais

Perguntas comparativas

4. Benchmark Expandido
Testar mais modelos (BERT-large, ALBERT, DeBERTa)

Comparar versões quantizadas

Avaliar trade-off tamanho vs. performance

Testar em diferentes hardwares (CPU, GPU, TPU)

5. Análise de Custo-Benefício
Custo computacional por inferência

Uso de memória

Tempo de resposta médio

Custo em cloud computing

📚 Referências Técnicas
Artigos Científicos
DistilBERT: Sanh et al. (2019) - "DistilBERT, a distilled version of BERT"

RoBERTa: Liu et al. (2019) - "RoBERTa: A Robustly Optimized BERT Pretraining Approach"

SQuAD: Rajpurkar et al. (2016) - "SQuAD: 100,000+ Questions for Machine Comprehension"

Documentação
Hugging Face Transformers: https://huggingface.co/docs/transformers

SQuAD Dataset: https://rajpurkar.github.io/SQuAD-explorer/

DBpedia Entity: https://huggingface.co/datasets/BeIR/dbpedia-entity

Links dos Modelos
DistilBERT SQuAD: https://huggingface.co/distilbert-base-cased-distilled-squad

RoBERTa SQuAD2: https://huggingface.co/deepset/roberta-base-squad2

📧 Informações do Projeto
Projeto: Análise Comparativa de Modelos de Question Answering

Dataset: DBpedia Entity Generated Queries (shard_055)

Amostra: 1000 exemplos

Modelos: DistilBERT vs. RoBERTa

Métricas: Score de confiança, Overlap, Diferenças

Timestamp: YYYYMMDD_HHMMSS

Ambiente: Google Colab com GPU T4

🎓 Como Contribuir
Para Extensão da Análise:
Testar com mais modelos do Hugging Face

Aplicar a diferentes datasets

Implementar métricas adicionais

Realizar análise de erro detalhada

Para Melhorias no Código:
Otimizar processamento em batch

Adicionar cache para resultados

Implementar paralelização

Criar visualizações interativas

Para Documentação:
Adicionar exemplos práticos

Incluir casos de uso específicos

Documentar limitações encontradas

Criar guias de deploy

⚡ Dicas Rápidas para Uso
Para Carregar os Resultados:
python
import pandas as pd
df = pd.read_csv('resultados_completos_YYYYMMDD_HHMMSS.csv')
Para Análise dos Extremos:
python
# Top 10 melhores do DistilBERT
top_distilbert = pd.read_csv('distilbert_top10_melhores_YYYYMMDD_HHMMSS.csv')

# Top 10 piores globais
piores_globais = pd.read_csv('global_top10_piores_YYYYMMDD_HHMMSS.csv')
Para Análise de Discordâncias:
python
if os.path.exists('discordancias_YYYYMMDD_HHMMSS.csv'):
    discordancias = pd.read_csv('discordancias_YYYYMMDD_HHMMSS.csv')
    print(f"Encontradas {len(discordancias)} discordâncias significativas")
📊 Glossário de Termos
QA (Question Answering): Sistema que responde perguntas baseado em contexto

Score de Confiança: Probabilidade atribuída pelo modelo à correção da resposta

Overlap: Porcentagem de palavras da resposta presentes no contexto

Alucinação: Quando o modelo gera informação não presente no contexto

SQuAD: Stanford Question Answering Dataset, benchmark para QA

Fine-tuning: Processo de adaptar um modelo pré-treinado para uma tarefa específica

Inferência: Processo de obter respostas do modelo

