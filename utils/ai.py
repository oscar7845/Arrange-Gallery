from . import file
from . import db
import face_recognition
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import pickle
import itertools
import os
import cv2


def detect_persons(
    df,
    tolerance=0.6,
    checkpoint_path="./data/tmp/detect_person_checkpoint.pkl",
    checkpoint_interval=10,
):
    j = 0
    unknown_counter = 0

    print("----- Face Comparison of detection started -----")
    print(f"Number of persons to comparison with each other : {len(df)}")
    print(f"Number of batches: {int(len(df)/checkpoint_interval)+1}")
    print(f"Number of checkpoint_interval: {checkpoint_interval}")

    try:
        with open(checkpoint_path, "rb") as f:
            df, j, unknown_counter = pickle.load(f)
            print(f"Checkpoint found, continuing from id {j}")
    except FileNotFoundError:
        pass

    for i, row in tqdm(itertools.islice(df.iterrows(), j, None), total=len(df)):
        known_face_names, known_face_encodings = db.get_known_face_encodings(df)

        face_encoding = row["face_encoding"]

        if face_encoding is None: 
            continue

        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding, tolerance=tolerance
        )

        if any(matches):
            name = known_face_names[np.argmax(matches)]
            df.loc[i, "id"] = name
        else:
            df.loc[i, "id"] = f"Unknown{unknown_counter}"
            unknown_counter += 1

        if i % checkpoint_interval == 0 and i != 0:
            print()
            print()
            print(f"  Checkpoint - {i+1}/{len(df)}")
            print()

            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            with open(checkpoint_path, "wb+") as f:
                pickle.dump((df, i, unknown_counter), f)

    print("----- Face comparison and person clustering performed on dataset -----")
    print(f" An amount of {unknown_counter} new persons were found!")

    return df


def multi_process_detect_faces(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(
        image, model="hog"
    )  
    face_encodings = face_recognition.face_encodings(image, face_locations)

    df = db.create(["image_path", "box", "face_encoding", "id"])

    if len(face_locations) == 0:
        new_row = pd.DataFrame(
            {
                "image_path": [image_path],
                "box": [None],
                "face_encoding": [None],
                "id": [None],
            }
        )
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        for rect, encodings in zip(face_locations, face_encodings):
            new_row = pd.DataFrame(
                {
                    "image_path": [image_path],
                    "box": [rect],
                    "face_encoding": [encodings],
                    "id": [None],
                }
            )
            df = pd.concat([df, new_row], ignore_index=True)

    return df

def multi_process_detect_all_faces_in_album(
    path,
    workers=8,
    checkpoint_path="./data/tmp/detect_faces_checkpoint.pkl",
    checkpoint_interval=2,
):
    image_paths = file.find_images(path)

    if len(image_paths) < checkpoint_interval:
        splitted_paths = [image_paths]
    else:
        splitted_paths = np.array_split(
            image_paths, len(image_paths) / checkpoint_interval
        )

    print("----- Face detection of album started -----")
    print(f"Number of images: {len(image_paths)}")
    print(f"Number of batches: {int(len(image_paths)/checkpoint_interval)+1}")
    print(f"Number of checkpoint_interval: {checkpoint_interval}")

    i = 0
    dfss = []
    try:
        with open(checkpoint_path, "rb") as f:
            dfss, i = pickle.load(f)
            print(f"Checkpoint found, continuing from id {i}")
    except FileNotFoundError:
        pass

    with Pool(workers) as pool:
        for j, path in enumerate(
            tqdm(itertools.islice(splitted_paths, i, None), total=len(splitted_paths)),
            start=i,
        ):
            dfs = []
            for result in tqdm(
                pool.imap(multi_process_detect_faces, path), total=len(path)
            ):
                dfs.append(result)

            try:
                dfss.append(pd.concat(dfs, ignore_index=True))
            except ValueError as ve:
                pass

            print()
            print()
            print(f"  Checkpoint - {j+1}/{len(splitted_paths)}")
            print()

            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            with open(checkpoint_path, "wb+") as f:
                pickle.dump((dfss, j + 1), f)

        pool.close()
        pool.join()

    try:
        df = pd.concat(dfss, ignore_index=True)
    except ValueError as ve:
        pass  

    print("")
    print(" ---- Statistics -----")
    print(f"A total of {len(df)} faces were detected in album.")
    print("")

    return df