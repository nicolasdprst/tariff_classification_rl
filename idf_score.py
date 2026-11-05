import pandas
import math
import re

def idf_score(text:str, dataFrame:pandas.DataFrame):
    pattern = rf"\b{re.escape(text)}\b"
    mask = dataFrame["Text"].astype(str).str.contains(pattern,case=False,regex=True)
    df_word = mask.sum()              #équivalent à (dataFrame.loc(mask)).shape[0]
    N = dataFrame.shape[0]
    return math.log((N+1) /(df_word+1) )     # +1 pour éviter la division par 0