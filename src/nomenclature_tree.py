"""On charge la nomenclature du fichier xlsx une fois pour tout et on construit les relations parent-enfants"""
from functools import lru_cache
from pathlib import Path
import re

import pandas as pd


WORKBOOK = Path(__file__).resolve().parents[1] / "data/reference/nomenclature.xlsx"


class NomenclatureTree:
    def __init__(self, path=WORKBOOK):
        frame = pd.read_excel(path, header=4, dtype=object)         #On lit le excel
        self.labels = {}
        self.display_codes = {}
        for _, row in frame.iterrows():
            raw = row.get("CN")
            if pd.isna(raw):
                continue
            code = re.sub(r"\s+", "", str(raw))
            if not code.isdigit() or len(code) not in (4, 6, 8):
                continue
            self.labels[code] = "" if pd.isna(row.get("Text")) else str(row["Text"])
            self.display_codes[code] = str(raw)
        self.parent = {}
        self.children = {code: [] for code in self.labels}
        # On relie chaque code à son parent direct : CN8 > CN6 > CN4 > chapitre.
        for code in self.labels:
            ancestors = [candidate for candidate in (code[:4], code[:6])
                         if candidate != code and candidate in self.labels]
            parent = max(ancestors, key=len) if ancestors else code[:2]
            self.parent[code] = parent
            self.children.setdefault(parent, []).append(code)
        self.paths = {}
        # On garde aussi le chemin complet pour présenter le contexte de chaque code au modèle.
        for code in self.labels:
            chain = []
            current = code
            while current in self.labels:
                chain.append(self.labels[current])
                current = self.parent[current]
            self.paths[code] = " > ".join(reversed(chain))

    def subcategories(self, code):
        # Les enfants sont déjà en mémoire : aucun nouveau parcours du fichier Excel.
        return [
            {"CN": self.display_codes[child], "Text": self.labels[child]}
            for child in self.children.get(code, ())
        ]


@lru_cache(maxsize=1)
def get_tree():
    # Le cache partage le même arbre entre tous les appels du processus.
    return NomenclatureTree()
