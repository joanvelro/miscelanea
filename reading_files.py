import pandas as pd
dfs = []
sqlall = "select * from mytable"
for chunk in pd.read_sql_query(query , connection, chunksize=10000):
    dfs.append(chunk)
my_df = pd.concat(dfs)