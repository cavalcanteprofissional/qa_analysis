import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from src.metrics_calculator import MetricsCalculator


def test_annotate_overlap_word_level():
    """Testa overlap palabra-a-palabra entre resposta e contexto."""
    df = pd.DataFrame([
        {"question": "q1", "context": "the cat sat on the mat", "model": "m1", "answer": "cat sat", "score": 0.9},
        {"question": "q1", "context": "the dog ran fast", "model": "m2", "answer": "dog", "score": 0.8},
        {"question": "q2", "context": "hello world test", "model": "m1", "answer": None, "score": 0.6},
    ])
    mc = MetricsCalculator()
    annotated = mc.annotate_overlap(df)
    
    # resposta 'cat sat' tem 2 palavras, ambas no contexto -> overlap 2/2 = 1.0
    row1 = annotated[annotated.model == "m1"].iloc[0]
    assert abs(float(row1.overlap) - 1.0) < 1e-6
    
    # resposta 'dog' tem 1 palavra, está no contexto -> overlap 1/1 = 1.0
    row2 = annotated[annotated.model == "m2"].iloc[0]
    assert abs(float(row2.overlap) - 1.0) < 1e-6
    
    # resposta faltante -> overlap 0
    row3 = annotated[(annotated.question == "q2") & (annotated.model == "m1")].iloc[0]
    assert float(row3.overlap) == 0.0


def test_partial_word_overlap():
    """Testa overlap parcial (nem todas as palavras da resposta no contexto)."""
    df = pd.DataFrame([
        {"question": "q", "context": "the quick brown fox", "model": "m1", "answer": "brown dog", "score": 0.9},
    ])
    mc = MetricsCalculator()
    annotated = mc.annotate_overlap(df)
    
    # resposta 'brown dog' tem 2 palavras, apenas 'brown' no contexto -> overlap 1/2 = 0.5
    row = annotated.iloc[0]
    assert abs(float(row.overlap) - 0.5) < 1e-6


def test_no_word_overlap():
    """Testa quando nenhuma palavra da resposta está no contexto."""
    df = pd.DataFrame([
        {"question": "q", "context": "the cat sat", "model": "m1", "answer": "dog bird", "score": 0.9},
    ])
    mc = MetricsCalculator()
    annotated = mc.annotate_overlap(df)
    
    # nenhuma palavra de 'dog bird' no contexto -> overlap 0/2 = 0.0
    row = annotated.iloc[0]
    assert float(row.overlap) == 0.0
