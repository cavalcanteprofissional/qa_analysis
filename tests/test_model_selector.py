from src.model_selector import ModelSelector


def test_list_available():
    ms = ModelSelector()
    avail = ms.list_available()
    assert "distilbert" in avail
    assert "bert" in avail


def test_select_models_all():
    ms = ModelSelector()
    sel = ms.select_models("all")
    assert isinstance(sel, list) and len(sel) >= 1
