import pandas as pd

def create(df_detection_column_names):
    return pd.DataFrame(columns=df_detection_column_names)

def save(df, store_csv_path):
    df.to_csv(store_csv_path, index=False)

def load(store_csv_path):
    return pd.read_csv(store_csv_path)

def get_all_ids(df):
    return df['id'].unique()

def replace_id(df,old_id,new_id):
    df['id'] = df['id'].replace(old_id, new_id)
    return df 

def get_all_images_of_individual(df,id):
    return df[df['id'] == id]["image_path"].unique()

def merge_ids(df, id1,id2):
    # Will merge two id:s to represent the same person and get id1:s name
    return replace_id(df, id2, id1)

def get_all_images_missing_faces(df):
    return df[df['box'] == None]["image_path"]