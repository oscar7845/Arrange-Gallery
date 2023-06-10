import face_recognition
from . import file
import pandas as pd
import numpy as np

def get_known_face_encodings(df):
    known_face_names = []
    known_face_encodings = []
    for index, row in df.iterrows():
        if row["id"] is not None:
            known_face_names.append(row["id"])
            known_face_encodings.append(row["face_encoding"])
    return known_face_names, known_face_encodings

def detect_persons(df, tolerance=0.6):
    #
    # Calculates matches between faces and updates dataframe with names
    #
    unknown_counter=0
    for i, row in df.iterrows():
        known_face_names, known_face_encodings = get_known_face_encodings(df)

        face_encoding = row["face_encoding"]

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)

        if any(matches):
            name = known_face_names[np.argmax(matches)]
            df.loc[i, 'id'] = name
        else:
            df.loc[i, 'id'] = f"Unknown{unknown_counter}"
            unknown_counter+=1
    return df

def detect_faces(image_path):
    # Load the image
    image = face_recognition.load_image_file(image_path)

    # Find all the faces in the image
    face_locations = face_recognition.face_locations(image)

    # Extract face encodings for all detected faces
    face_encodings = face_recognition.face_encodings(image, face_locations)

    return face_locations, face_encodings


def detect_all_faces_in_album(path):
    #
    # Loops over each image in path subdirectories and detects faces and calculates face encodings. Returns a pandas.DataFrame where each detected face is a row.
    #
    image_paths = file.find_images(path)

    df_detection_column_names = ["image_path", "box", "face_encoding", "id"]
    df = pd.DataFrame(columns=df_detection_column_names)

    for i,path in enumerate(image_paths):

        print(f"On image {i+1}/{len(image_paths)}")
        face_rects, face_encodings = detect_faces(path)

        print(f"In image {path} {len(face_rects)} faces detected.")

        
        for  rect, encodings in zip(face_rects,face_encodings):
            new_row = pd.DataFrame({"image_path": [path], "box": [rect], "face_encoding": [encodings],"id": [None]})
            df = pd.concat([df,new_row], ignore_index=True)
    
    print("")
    print(" ---- Statistics -----")
    print(f"A total of {len(df)} faces were detected in album.")
    print("Showing DataFrame:")
    print(df)
    print("")

    return df 