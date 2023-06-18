import nvtabular as nvt
import pandas as pd
import os


INPUT_DATA_DIR = '..'
sessions_gdf = pd.read_parquet(os.path.join(INPUT_DATA_DIR, "processed_nvt/part_0.parquet"))
print(sessions_gdf.head(2))

session_ids = pd.read_parquet('categories/unique.session_id.parquet')
item_ids = pd.read_parquet('categories/unique.prev_items.parquet')

# First, make sure the index of `session_ids` is integer type, this ensures the matching works correctly
session_ids.index = session_ids.index.astype(int)

# Then map `session_id` in `sessions_df` to index in `session_ids`
sessions_gdf['session_org'] = sessions_gdf['session_id'].map(session_ids['session_id'])

sessions_df_test = sessions_gdf[sessions_gdf['session_org'] < 999999]
sessions_gdf = sessions_gdf.drop('session_org', axis=1)
sessions_df_test = sessions_df_test.drop('session_org', axis=1)

sessions_df_test.to_parquet('../processed_nvt/test_0.parquet')
print(len(sessions_gdf))








