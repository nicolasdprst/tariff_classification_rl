import pandas as pd

df_nomenclature = pd.read_excel("nomenclature.xlsx", header=4)  #3 colonnes : Texte, CN et Unit

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
    dataFrame.insert(0,"score",0)
    query = string.strip()    #on supprime les espaces inutiles en début et en fin

    query_words = query.split()

    for i in range(len(query_words)):
        mask = (
            dataFrame["Text"]
            .astype(str)
            .str.contains(query_words[i], case=False, na=False, regex=False)
            |
            dataFrame["CN"]
            .astype(str)
            .str.contains(query_words[i], case=False, na=False, regex=False)
        )
        dataFrame.loc[mask, ["score"]] += 1
    dataFrameFiltered = dataFrame[dataFrame["score"] > 0]
    return dataFrameFiltered.sort_values(by="score", ascending=False)

def search_nomenclature_tool(query: str, limit: int = 10):
    df = search_nomenclature(query)

    results = df.head(limit)

    return results.fillna("").astype(str).to_dict(orient="records")


if __name__ == "__main__":
    query = "white chocolate containing cocoa"
    print(search_nomenclature(query))