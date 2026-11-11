import pandas as pd
from pathlib import Path

NOMENCLATURE_PATH = Path(__file__).resolve().parents[1] / "data" / "reference" / "nomenclature.xlsx"
df_nomenclature = pd.read_excel(NOMENCLATURE_PATH, header=4)  #3 colonnes : Texte, CN et Unit

def get_valid_codes() -> set[str]:
    codes = df_nomenclature["CN"]

    codes = codes.astype(str)
    codes = codes.str.replace(r"\s+", "", regex=True)    #On enlève les espaces dans les codes pour avoir un bon format

    mask = codes.str.fullmatch(r"\d{8}")

    valid_codes = codes[mask]                             #on garde que les codes qui sont de taille 8

    return set(valid_codes)

    

if __name__ == "__main__":
    valid_codes = get_valid_codes()
    print(f"Nombre de codes cvorrects : {len(valid_codes)}")
    print("87031018" in valid_codes)