import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from src.data_loader import ShardLoader
from pathlib import Path


def test_discover_shards():
    sl = ShardLoader("data/shards")
    assert isinstance(sl.available_shards, list)


def test_select_all():
    sl = ShardLoader("data/shards")
    sel = sl.select_shards("all")
    assert len(sel) == len(sl.available_shards)
