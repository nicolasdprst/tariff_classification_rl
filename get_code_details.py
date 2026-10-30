import pandas as pd

df_nomenclature = pd.read_excel("nomenclature.xlsx", header=4)

def get_code_details(string: str):

    query = string.strip()
    mask = (
        df_nomenclature["CN"]
        .astype(str)
        .str.contains(query, case=False, na=False, regex=False)
    )
    results = df_nomenclature.loc[mask,["CN","Text","Unit"]]
    return results

if __name__ == "__main__":
    print(get_code_details("0101"))