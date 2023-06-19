import pandas as pd

def load_dataframe(path):
    return pd.read_csv(path)

def create(df_detection_column_names):
    return pd.DataFrame(columns=df_detection_column_names)

def save(df, store_csv_path):
    df.to_csv(store_csv_path, index=False)

def load(store_csv_path):
    return pd.read_csv(store_csv_path)

def get_all_ids(df):
    return df["id"].unique()

def replace_id(df, old_id, new_id):
    df["id"] = df["id"].replace(old_id, new_id)
    return df

def get_all_images_of_individual(df, id):
    return df[df["id"] == id]["image_path"].unique()

def get_all_occurrences_of_individual(df, id):
    return df[df["id"] == id]

def get_all_images_of_non_individuals(df):
    return df[df["id"].isnull()]["image_path"].unique()

def get_all_images_missing_faces(df):
    return df[df["box"] == None]["image_path"].unique()

# Take dataframe and return two lists of found person name and corresponding face encoding
def get_known_face_encodings(df):
    known_face_names = []
    known_face_encodings = []
    for _, row in df.iterrows():
        if row["id"] is not None:
            known_face_names.append(row["id"])
            known_face_encodings.append(row["face_encoding"])
    return known_face_names, known_face_encodings