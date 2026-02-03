#!/usr/bin/env bash
poetry run python -m src.main --shards all --models all
poetry run python -m src.main --shards  shard_055.csv --models distilbert roberta