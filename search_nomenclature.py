import pandas as pd

df_nomenclature = pd.read_excel("nomenclature.xlsx", header=4)  #3 colonnes : Texte, CN et Unit

def search_nomenclature(string : str):
    """Recherche une chaîne de caractère dans la nomenclature

    args
        string (str): la chaîne que l'on souhaite rechercher
    
    returns

    """
    query = string.strip()    #on supprime les espaces inutiles en début et en fin

    mask = (
        df_nomenclature["Text"]
        .astype(str)
        .str.contains(query, case=False, na=False, regex=False)
        |
        df_nomenclature["CN"]
        .astype(str)
        .str.contains(query, case=False, na=False, regex=False)
    )

    results = df_nomenclature.loc[mask, ["CN","Text","Unit"]]
    return results

def search_nomenclature_tool(query: str, limit: int = 10):
    df = search_nomenclature(query)

    results = df.head(limit)

    return results.fillna("").astype(str).to_dict(orient="records")


if __name__ == "__main__":
    print(search_nomenclature("horses"))
    print(search_nomenclature("0101"))
    print(search_nomenclature_tool("chocolate"))