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


def create_face_collage(df, persons, target_path, resolution):
    print("----- Generate Collage -----")
    cropped_faces = []
    for person in persons:
        personal_df = db.get_all_occurrences_of_individual(df, person)

        for _, row in personal_df.iterrows():
            img = cv2.imread(row["image_path"])
            top, right, bottom, left = row["box"]
            width = right - left
            height = bottom - top
            x = left
            y = top
            cropped_faces.append(img[y : y + height, x : x + width])

    merged_image = merge_images(cropped_faces, resolution[0], resolution[1])
    if merged_image is None:
        print(f"No faces found for person {person}.")
        return None

    os.makedirs(target_path, exist_ok=True)
    dest = file.get_appropriate_incremental_name("face_collage.png", target_path)
    cv2.imwrite(dest, merged_image)
    print(f"Saved collage at {dest}.")

    return merged_image


def merge_images(
    images, output_width, output_height
):  
    if len(images) == 0:
        return None

    num_images = len(images)
    num_cols = int(np.ceil(np.sqrt(num_images)))
    num_rows = int(np.ceil(num_images / num_cols))

    print(f"Amount of images to merge: {len(images)}")
    print(f"Image merge dimensions: {num_cols}x{num_rows}")
    print(f"Output image resolutiong: {output_width}x{output_height}")

    subimage_width = int(output_width / num_cols)
    subimage_height = int(output_height / num_rows)

    output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        row_idx = i // num_cols
        col_idx = i % num_cols

        img_resized = cv2.resize(img, (subimage_width, subimage_height))

        x = col_idx * subimage_width
        y = row_idx * subimage_height

        output_image[y : y + subimage_height, x : x + subimage_width, :] = img_resized

    return output_image

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