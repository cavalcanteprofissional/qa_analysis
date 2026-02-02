import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from src.metrics_calculator import MetricsCalculator


def test_annotate_overlap_basic():
    df = pd.DataFrame([
        {"question": "q1", "context": "c1", "model": "m1", "answer": "A", "score": 0.9},
        {"question": "q1", "context": "c1", "model": "m2", "answer": "A", "score": 0.8},
        {"question": "q1", "context": "c1", "model": "m3", "answer": "B", "score": 0.7},
        {"question": "q2", "context": "c2", "model": "m1", "answer": None, "score": 0.6},
        {"question": "q2", "context": "c2", "model": "m2", "answer": "C", "score": 0.5},
    ])
    mc = MetricsCalculator()
    annotated = mc.annotate_overlap(df)
    # for q1/c1: answers A appear twice -> overlap_count 2, fraction 2/3
    row_m1 = annotated[(annotated.model == "m1") & (annotated.question == "q1")].iloc[0]
    assert int(row_m1.overlap_count) == 2
    assert abs(float(row_m1.overlap_fraction) - (2/3)) < 1e-6
    # for missing answer row overlap_count should be 0
    missing_row = annotated[(annotated.question == "q2") & (annotated.model == "m1")].iloc[0]
    assert int(missing_row.overlap_count) == 0
    assert float(missing_row.overlap_fraction) == 0.0


def test_per_model_metrics_include_overlap():
    df = pd.DataFrame([
        {"question": "q", "context": "c", "model": "m1", "answer": "X", "score": 0.9},
        {"question": "q", "context": "c", "model": "m2", "answer": "X", "score": 0.8},
        {"question": "q", "context": "c", "model": "m3", "answer": "Y", "score": 0.7},
    ])
    mc = MetricsCalculator()
    metrics = mc.calculate_all_metrics(df)
    per = metrics["per_model"]
    # m1 and m2 should have avg_overlap_fraction 2/3, m3 should have 1/3
    assert abs(per["m1"]["avg_overlap_fraction"] - (2/3)) < 1e-6
    assert abs(per["m3"]["avg_overlap_fraction"] - (1/3)) < 1e-6


def test_overall_metrics_include_overlap():
    df = pd.DataFrame([
        {"question": "q", "context": "c", "model": "m1", "answer": "A", "score": 0.9},
        {"question": "q", "context": "c", "model": "m2", "answer": "A", "score": 0.8},
        {"question": "q", "context": "c", "model": "m3", "answer": "B", "score": 0.7},
    ])
    mc = MetricsCalculator()
    metrics = mc.calculate_all_metrics(df)
    overall = metrics["overall"]
    # overlap fractions are [2/3, 2/3, 1/3] -> mean = (2/3+2/3+1/3)/3 = 7/9 ~ 0.777...
    expected = (2/3 + 2/3 + 1/3) / 3
    assert abs(overall["avg_overlap_fraction"] - expected) < 1e-6
