from pathlib import Path

from src.idf_score import idf_score
import pandas as pd
import re


NOMENCLATURE_PATH = Path(__file__).resolve().parents[1] / "data" / "reference" / "nomenclature.xlsx"
df_nomenclature = pd.read_excel(NOMENCLATURE_PATH, header=4)  #3 colonnes : Texte, CN et Unit

def search_nomenclature(string : str):
    """Recherche une chaîne de caractère dans la nomenclature

    args
        string (str): la chaîne que l'on souhaite rechercher
    
    returns
        pandas.DataFrame : contenant une colonne de plus avec un score,
                            indiquant le nombre de mots de la requete trouves dans le texte,
                            et trie par ordre decroissant du score
    """

    dataFrame = df_nomenclature.copy()
    dataFrame.insert(0,"score",0.0)
    query = string.strip()    #on supprime les espaces inutiles en début et en fin

    query_words = query.split()

    for i in range(len(query_words)):
        pattern = rf"\b{re.escape(query_words[i])}\b"
        mask = (
            dataFrame["Text"]
            .astype(str)
            .str.contains(pattern, case=False, na=False, regex=True)
            |
            dataFrame["CN"]
            .astype(str)
            .str.contains(query_words[i], case=False, na=False, regex=False)
        )
        dataFrame.loc[mask, ["score"]] += idf_score(query_words[i],dataFrame)

    dataFrameFiltered = dataFrame[dataFrame["score"] > 0]
    return dataFrameFiltered.sort_values(by="score", ascending=False)

def search_nomenclature_tool(query: str, limit: int = 5):
    df = search_nomenclature(query)
    df.drop(columns=["New/reused","Unit","score"], inplace=True)    #colonnes inutiles
    results = df.head(limit)
    return results.fillna("").astype(str).to_dict(orient="records")

if __name__ == "__main__":
    query = "white chocolate containing cocoa"
    print(search_nomenclature_tool(query))
