# %%
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
plt.close('all')

# %%
frames = []
import traceback
for txt in os.listdir('./aol')[0:1]:
    try:
        queries = pd.read_csv(f'./aol/{txt}', sep='\t', dtype={"AnonID": "Int64", "Query": "string", "QueryTime": "string", "ItemRank": "Int32", "ClickURL": "string"}, keep_default_na=False, na_values=[""])
        frames.append(queries)
    except Exception as e:
        print(txt)
        print(e)
        print(e.args)
        traceback.print_exc()

queries = pd.concat(frames)
queries['QueryTime'] = pd.to_datetime(queries['QueryTime'])
queries.insert(5, 'SessionNum', 0)
queries.insert(6, 'Type', '')

# %%
grouped_by_id = queries.groupby('AnonID')
qqq = grouped_by_id['Query'].nunique()
ppp = qqq.sort_values(ascending=False)

filtered = ppp.index


# %%
def dict_from_row(row, logType, sessionNum):
    return {
      'AnonID': row['AnonID'],
      'Query': row['Query'],
      'QueryTime': row['QueryTime'],
      'Type': logType,
      'SessionNum': sessionNum,
      'ItemRank': row['ItemRank'] if logType == "Click" else np.nan,
      'ClickURL': row['ClickURL'] if logType == "Click" else np.nan
    }

new_logs = []
for id in filtered:
    user_log = queries.loc[queries['AnonID'] == id]
    current_query = ''
    current_query_time = ''
    current_session_num = 0
    for i in range(len(user_log)):
        if user_log.iat[i, 2] >= current_query_time + pd.Timedelta(30, unit="min"):
            current_session_num += 1
            current_query = user_log.iat[i, 1]
            current_query_time = user_log.iat[i, 2]

            new_logs.append(dict_from_row(user_log.iloc[i], "Query", current_session_num))

            if pd.notna(user_log.iat[i, 4]):
                new_logs.append(dict_from_row(user_log.iloc[i], "Click", current_session_num))

        else:
            # Within the same sessoin
            if (user_log.iat[i, 1] == current_query) and (user_log.iat[i, 2] > current_query_time):
                # pagination case
                current_query_time = user_log.iat[i, 2]
                new_logs.append(dict_from_row(user_log.iloc[i], "NextPage", current_session_num))

                if pd.notna(user_log.iat[i, 4]):
                    new_logs.append(dict_from_row(user_log.iloc[i], "Click", current_session_num))
            elif (user_log.iat[i, 1] == current_query) and (user_log.iat[i, 2] == current_query_time):
                # Same query, same timestamp ==> must be different clicks
                if pd.notna(user_log.iat[i, 4]):
                    new_logs.append(dict_from_row(user_log.iloc[i], "Click", current_session_num))
                else:
                    # Another very quick nextpage
                    new_logs.append(dict_from_row(user_log.iloc[i], "NextPage", current_session_num))
            elif (user_log.iat[i, 1] != current_query) :
                ## Query refinement
                current_query_time = user_log.iat[i, 2]
                current_query = user_log.iat[i, 1]
                new_logs.append(dict_from_row(user_log.iloc[i], "Query", current_session_num))

                if pd.notna(user_log.iat[i, 4]):
                    ## If user clicked sth from query refinement
                    new_logs.append(dict_from_row(user_log.iloc[i], "Click", current_session_num))
        


new_logs_df = pd.DataFrame(new_logs)
new_logs_df.to_csv('./new_logs.csv', index_label="idx")



