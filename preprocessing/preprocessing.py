import os
import numpy as np
import pandas as pd
from functools import lru_cache
import gc
import shutil
import glob
import nvtabular as nvt
from merlin.dag import ColumnSelector
from merlin.schema import Schema, Tags



train_data_dir = '../data/'
test_data_dir = '../data/'
task = 'task1'
PREDS_PER_SESSION = 100

SESSIONS_MAX_LENGTH = 10
MINIMUM_SESSION_LENGTH = 2


def read_product_data():
    return pd.read_csv(os.path.join(train_data_dir, 'products_train.csv'))
def read_train_data():
    return pd.read_csv(os.path.join(train_data_dir, 'sessions_train.csv'))
def read_test_data(task):
    return pd.read_csv(os.path.join(test_data_dir, f'sessions_test_{task}.csv'))
def str2list(x):
    x = x.replace('[', '').replace(']', '').replace("'", '').replace('\n', ' ').replace('\r', ' ')
    l = [i for i in x.split() if i]
    return l
def relative_price_to_avg_categ(col, gdf):
    epsilon = 1e-5
    col = ((gdf['price'] - col) / (col + epsilon)) * (col > 0).astype(int)
    return col



if __name__=='__main__':
    products = read_product_data()
    train_sessions = read_train_data()
    test_sessions = read_test_data(task)

    # print(products.sample(5))
    # print(train_sessions.head(5))
    # print(test_sessions.head(5))

    products_DE = products[products['locale'] == 'DE']
    train_sessions_DE = train_sessions[train_sessions['locale'] == 'DE']
    test_sessions_DE = test_sessions[test_sessions['locale'] == 'DE']
    # print(test_sessions_DE.shape)  # in 14
    # print(train_sessions_DE.head())

    # %%
    dat_train = train_sessions_DE['prev_items'].tolist()
    dat_test = test_sessions_DE['prev_items'].tolist()

    cleaned_train = []
    for i in range(len(dat_train)):
        cleaned_train.append(str2list(dat_train[i]))
    # cleaned_train

    cleaned_test = []
    for i in range(len(dat_test)):
        cleaned_test.append(str2list(dat_test[i]))
    # cleaned_test

    train_sessions_DE['prev_items'] = cleaned_train
    test_sessions_DE['prev_items'] = cleaned_test

    min_count = test_sessions_DE['prev_items'].apply(lambda x: len(x)).min()
    train_sessions_DE['prev_items'] = train_sessions_DE.apply(lambda row: row['prev_items'] + [row['next_item']],axis=1)
    train_sessions_DE.reset_index(level=0, inplace=True)
    train_sessions_DE.rename(columns={'index':'session_id'}, inplace=True)
    # Split 'prev_items' into separate rows
    train_sessions_DE = train_sessions_DE.explode('prev_items')

    test_sessions_DE.reset_index(level=0, inplace=True)
    test_sessions_DE.rename(columns={'index': 'session_id'}, inplace=True)
    # Split 'prev_items' into separate rows
    test_sessions_DE = test_sessions_DE.explode('prev_items')

    # print(train_sessions_DE.head())  # in 28
    train_sessions_DE['session_id'] = train_sessions_DE['session_id'] + 1000000
    # print(test_sessions_DE.head())  # differ from src.

    session_attribute_train = pd.merge(train_sessions_DE, products_DE, left_on='prev_items', right_on='id',
                                       how='left').drop(['id', 'locale_x', 'locale_y'], axis=1)
    session_attribute_test = pd.merge(test_sessions_DE, products_DE, left_on='prev_items', right_on='id',
                                      how='left').drop(['id', 'locale_x', 'locale_y'], axis=1)
    
    # print(session_attribute_train.head(5))
    # print(session_attribute_test.head(5))

    raw_df1 = session_attribute_train.drop('next_item', axis=1)
    # print(np.min(raw_df1.session_id))
    # print(np.max(session_attribute_test['session_id']))
    # print(session_attribute_test.shape)  # (450090, 11), in 41

    join_data = pd.concat([raw_df1, session_attribute_test])
    raw_df = join_data
    # print(raw_df.head())


    ###################################################
    cols = list(raw_df.columns)
    cols.remove('session_id')
    print(cols)

    # load data
    df_event = nvt.Dataset(raw_df)

    # categorify user_session
    cat_feats = ['session_id'] >> nvt.ops.Categorify()

    workflow = nvt.Workflow(cols + cat_feats)
    workflow.fit(df_event)
    df = workflow.transform(df_event).to_ddf().compute()

    item_id = ['prev_items'] >> nvt.ops.TagAsItemID()
    cat_feats = item_id + ['title', 'price', 'brand', 'color', 'size', 'model', 'material', 'author',
                           'desc'] >> nvt.ops.Categorify(start_index=1)

    # print(df.head())
    # print(raw_df.head())

    price_log = ['price'] >> nvt.ops.LogOp() >> nvt.ops.Normalize(out_dtype=np.float32) >> nvt.ops.Rename(
        name='price_log_norm')
    avg_category_id_pr = ['brand'] >> nvt.ops.JoinGroupby(cont_cols=['price'], stats=["mean"]) >> nvt.ops.Rename(
        name='avg_category_id_price')
    relative_price_to_avg_category = (
            avg_category_id_pr >>
            nvt.ops.LambdaOp(relative_price_to_avg_categ, dependency=['price']) >>
            nvt.ops.Rename(name="relative_price_to_avg_categ_id") >>
            nvt.ops.AddMetadata(tags=[Tags.CONTINUOUS])
    )
    groupby_feats = ['session_id'] + cat_feats + price_log + relative_price_to_avg_category
    print(groupby_feats)

    # Define Groupby Workflow
    groupby_features = groupby_feats >> nvt.ops.Groupby(
        groupby_cols=["session_id"],
        aggs={
            'prev_items': ["list", "count"],
            'title': ["list"],
            'price_log_norm': ["list"],
            'relative_price_to_avg_categ_id': ["list"],
            'brand': ["list"],
            'color': ["list"],
            'size': ["list"],
            'model': ["list"],
            'material': ["list"],
            'author': ["list"],
            'desc': ["list"]
        },
        name_sep="-"
    )
    print(groupby_features)

    groupby_features_list = groupby_features[
        'prev_items-list',
        'title-list',
        'price_log_norm-list',
        'relative_price_to_avg_categ_id-list',
        'brand-list',
        'color-list',
        'size-list',
        'model-list',
        'material-list',
        'author-list',
        'desc-list'
    ]

    groupby_features_trim = groupby_features_list >> nvt.ops.ListSlice(-SESSIONS_MAX_LENGTH)
    sess_id = groupby_features['session_id'] >> nvt.ops.AddMetadata(tags=[Tags.CATEGORICAL])
    selected_features = sess_id + groupby_features['prev_items-count'] + groupby_features_trim
    filtered_sessions = selected_features >> nvt.ops.Filter(
        f=lambda df: df["prev_items-count"] >= MINIMUM_SESSION_LENGTH
    )
    print(filtered_sessions)

    workflow = nvt.Workflow(filtered_sessions)
    dataset = nvt.Dataset(df)

    workflow.fit_transform(dataset).to_parquet("../processed_nvt")
    print(workflow.output_schema)

