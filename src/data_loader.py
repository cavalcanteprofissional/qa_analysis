"""Data loader para shards CSV com seletor flexível."""
from pathlib import Path
from typing import List, Union
import pandas as pd
import glob
import fnmatch


class ShardLoader:
    def __init__(self, shards_dir: str):
        self.shards_dir = Path(shards_dir)
        self.available_shards = self._discover_shards()

    def _discover_shards(self) -> List[Path]:
        """Descobre todos os arquivos CSV no diretório de shards."""
        pattern = str(self.shards_dir / "*.csv")
        files = [Path(p) for p in glob.glob(pattern)]
        files.sort()
        return files

    def select_shards(self, selection: Union[str, List[str]]) -> List[Path]:
        """Seleciona shards baseado em:
        - "all": todos os shards
        - padrão com wildcard (ex: "shard_0*.csv")
        - lista específica de nomes
        """
        if not selection:
            return []

        if isinstance(selection, str):
            if selection == "all":
                return self.available_shards
            # single pattern
            pattern = selection
            matched = [p for p in self.available_shards if fnmatch.fnmatch(p.name, pattern)]
            return matched

        # list
        selected = []
        for item in selection:
            if item == "all":
                return self.available_shards
            matched = [p for p in self.available_shards if p.name == item or fnmatch.fnmatch(p.name, item)]
            selected.extend(matched)

        # unique and sorted
        uniq = sorted(list({p for p in selected}))
        return uniq

    def load_selected_shards(self, selection: Union[str, List[str]]) -> pd.DataFrame:
        """Carrega e concatena shards selecionados em um único DataFrame."""
        shards = self.select_shards(selection)
        if not shards:
            return pd.DataFrame()

        dfs = []
        for p in shards:
            try:
                df = pd.read_csv(p)
                df["_shard"] = p.name
                dfs.append(df)
            except Exception:
                continue

        if not dfs:
            return pd.DataFrame()

        out = pd.concat(dfs, ignore_index=True)
        return out
