import csv
import pandas as pd
import os
import numpy as np

frames = []
import traceback
for txt in os.listdir('./aol')[0:1]:
    try:
        queries = pd.read_csv(f'./aol/{txt}', sep='\t', dtype={"AnonID": "Int64", "Query": "string", "QueryTime": "string", "ItemRank": "Int32", "ClickURL": "string"})
        frames.append(queries)
    except Exception as e:
        print(txt)
        print(e)
        print(e.args)
        traceback.print_exc()

queries = pd.concat(frames)
queries['QueryTime'] = pd.to_datetime(queries['QueryTime'])

grouped_by_id = queries.groupby('AnonID')

qqq = grouped_by_id['Query'].nunique()
ppp = qqq.sort_values(ascending=False)

filtered = ppp.index