from . import file
from . import db
import face_recognition
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import pickle
import os
import itertools

def get_known_face_encodings(df):
    known_face_names = []
    known_face_encodings = []
    for _, row in df.iterrows():
        if row["id"] is not None:
            known_face_names.append(row["id"])
            known_face_encodings.append(row["face_encoding"])
    return known_face_names, known_face_encodings


def detect_persons(df, tolerance=0.6, checkpoint_name="./data/tmp/detect_person_checkpoint.pkl", checkpoint_interval=10):
    j = 0
    unknown_counter=0
    try:
        with open(checkpoint_name, "rb") as f:
            df,j,unknown_counter = pickle.load(f)
            print(f"Checkpoint found, continuing from id {j}")
    except FileNotFoundError:
        pass 

    for i, row in tqdm(itertools.islice(df.iterrows(), j, None),total=len(df)):
        known_face_names, known_face_encodings = get_known_face_encodings(df)

        face_encoding = row["face_encoding"]

        if face_encoding is None: # No faces found in image
            continue

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)

        if any(matches):
            name = known_face_names[np.argmax(matches)]
            df.loc[i, 'id'] = name
        else:
            df.loc[i, 'id'] = f"Unknown{unknown_counter}"
            unknown_counter+=1

        if i % checkpoint_interval == 0 and i != 0:
            with open(checkpoint_name, "wb+") as f:
                pickle.dump((df, i, unknown_counter), f)

    os.remove(checkpoint_name)

    return df

def detect_all_faces_in_album(path, workers=8,checkpoint_interval=50):
    return multi_process_detect_all_faces_in_album(path, workers=workers,checkpoint_interval=checkpoint_interval)

def multi_process_detect_faces(image_path):
    # Detects faces and face encoding using (HOG default) + Linear SVM face detection.
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image, model="hog") # Change model here! HOG is faster on CPU! CNN is more accurate and faster on GPU with CUDA!
    face_encodings = face_recognition.face_encodings(image, face_locations)

    df = db.create(["image_path", "box", "face_encoding", "id"])

    if len(face_locations) == 0:
        new_row = pd.DataFrame({"image_path": [image_path], "box": [None], "face_encoding": [None],"id": [None]})
        df = pd.concat([df,new_row], ignore_index=True)
    else:
        for rect, encodings in zip(face_locations,face_encodings):
            new_row = pd.DataFrame({"image_path": [image_path], "box": [rect], "face_encoding": [encodings],"id": [None]})
            df = pd.concat([df,new_row], ignore_index=True)

    return df

def multi_process_detect_all_faces_in_album(path, workers=8, checkpoint_name="./data/tmp/detect_faces_checkpoint.pkl", checkpoint_interval=2):
    # Loops over each image in path subdirectories and detects faces and calculates face encodings. Returns a pandas.DataFrame where each detected face is a row.
    # Multiprocess variant. Now also with checkpoint security, useful for long running tasks.
    image_paths = file.find_images(path)

    if  len(image_paths) < checkpoint_interval:
        splitted_paths = [image_paths]
    else:    
        splitted_paths = np.array_split(image_paths,len(image_paths)/checkpoint_interval)

    i = 0
    dfss = []
    try:
        with open(checkpoint_name, "rb") as f:
            dfss,i = pickle.load(f)
            print(f"Checkpoint found, continuing from id {i}")
    except FileNotFoundError:
        pass 

    with Pool(workers) as pool:
        for j,path in  enumerate(tqdm(itertools.islice(splitted_paths, i, None),total=len(splitted_paths))):
            
            dfs = []
            for result in tqdm(pool.imap(multi_process_detect_faces, path), total=len(path)):
                dfs.append(result)

            try:
                dfss.append(pd.concat(dfs, ignore_index=True))
            except ValueError as ve:
                pass # This means no images were found!
            
            print()
            print()
            print(f"  Checkpoint - {j+i+1}/{len(splitted_paths)}")
            print()
            with open(checkpoint_name, "wb+") as f:
                pickle.dump((dfss, j+i+1), f)

        pool.close()
        pool.join()

    try:
        df = pd.concat(dfss, ignore_index=True)
    except ValueError as ve:
        pass 

    os.remove(checkpoint_name)

    print("")
    print(" ---- Statistics -----")
    print(f"A total of {len(df)} faces were detected in album.")
    print("")

    return df 


"""
#TODO: for later
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

def sentiment_analysis(image_path):
    # Load the VGG-16 model
    model = tf.keras.applications.VGG16(weights='imagenet')
    
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    
    # Use the model to predict the emotions conveyed by the image
    preds = model.predict(x)
    results = decode_predictions(preds, top=5)[0]
    
    # Print the top 5 emotions predicted by the model
    for result in results:
        print(result[1], ':', result[2])

"""