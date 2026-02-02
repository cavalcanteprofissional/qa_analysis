import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model_selector import ModelSelector


def test_list_available():
    ms = ModelSelector()
    avail = ms.list_available()
    assert "distilbert" in avail


def test_select_models_all():
    ms = ModelSelector()
    sel = ms.select_models("all")
    assert isinstance(sel, list) and len(sel) >= 1
