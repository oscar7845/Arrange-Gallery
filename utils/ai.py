import face_recognition
from . import file
import pandas as pd
import numpy as np
import cv2
import time
from multiprocessing import Pool
from tqdm import tqdm

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
    # Calculates matches between faces and updates DataFrame with names
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

def single_process_detect_faces(image_path, rescale):
    #
    # Detects faces and face encoding using (HOG default) + Linear SVM face detection.
    #
    image = face_recognition.load_image_file(image_path)
    if rescale is not None:
        width,height,_ = image.shape
        new_size = (int(width*rescale), int(height*rescale))
        image = cv2.resize(image, new_size)
    face_locations = face_recognition.face_locations(image) # model can be changed!
    face_encodings = face_recognition.face_encodings(image, face_locations)

    return face_locations, face_encodings, image

def detect_all_faces_in_album(path, multi_process=True, workers=8, show_images=False, images_show_timer=1, rescale=None, estimate_time=True):
    if multi_process:
        return multi_process_detect_all_faces_in_album(path, workers=workers)
    else:
        return single_process_detect_all_faces_in_album(path, workers=workers, show_images=show_images, images_show_timer=images_show_timer, rescale=rescale, estimate_time=estimate_time)


def multi_process_detect_faces(image_path):
    #
    # Detects faces and face encoding using (HOG default) + Linear SVM face detection.
    #
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image) # model can be changed! cnn is CUDA accelerated!
    face_encodings = face_recognition.face_encodings(image, face_locations)

    df_detection_column_names = ["image_path", "box", "face_encoding", "id"]
    df = pd.DataFrame(columns=df_detection_column_names)

    for rect, encodings in zip(face_locations,face_encodings):
        new_row = pd.DataFrame({"image_path": [image_path], "box": [rect], "face_encoding": [encodings],"id": [None]})
        df = pd.concat([df,new_row], ignore_index=True)

    #print(f"Done with {image_path}")

    return df

def multi_process_detect_all_faces_in_album(path, workers=8):
    #
    # Loops over each image in path subdirectories and detects faces and calculates face encodings. Returns a pandas.DataFrame where each detected face is a row.
    # Multiprocess variant.
    #
    image_paths = file.find_images(path)

    with Pool(workers) as pool:
        dfs = []
        for result in tqdm(pool.imap(multi_process_detect_faces, image_paths), total=len(image_paths)):
            dfs.append(result)

        df = pd.concat(dfs, ignore_index=True)

        print("")
        print(" ---- Statistics -----")
        print(f"A total of {len(df)} faces were detected in album.")
        print("")

        pool.close()
        pool.join()

    return df 

def single_process_detect_all_faces_in_album(path, workers=8, show_images=False, images_show_timer=1, rescale=None, estimate_time=True):
    #
    # Loops over each image in path subdirectories and detects faces and calculates face encodings. Returns a pandas.DataFrame where each detected face is a row.
    #
    image_paths = file.find_images(path)

    df_detection_column_names = ["image_path", "box", "face_encoding", "id"]
    df = pd.DataFrame(columns=df_detection_column_names)
    

    for i,path in enumerate(image_paths):
        if estimate_time:
            start_time = time.time()

        print(f"On image {i+1}/{len(image_paths)}")
        face_rects, face_encodings, image = single_process_detect_faces(path, rescale)

        if show_images:
            for top, right, bottom, left in face_rects:
                cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

            resized_image = cv2.resize(image, (500, 500))
            cv2.imshow(path, resized_image)
            cv2.waitKey(images_show_timer)
            cv2.destroyAllWindows()

        print(f"In image {path} {len(face_rects)} faces detected.")
 
        for  rect, encodings in zip(face_rects,face_encodings):
            new_row = pd.DataFrame({"image_path": [path], "box": [rect], "face_encoding": [encodings],"id": [None]})
            df = pd.concat([df,new_row], ignore_index=True)
        
        if estimate_time:
            end_time = time.time()
            execution_time = (end_time - start_time)*(len(image_paths)-i+1)
            execution_time_str = time.strftime('%H:%M:%S', time.gmtime(execution_time))

            print(f" Estimated Time Left : {execution_time_str}")
    
    print("")
    print(" ---- Statistics -----")
    print(f"A total of {len(df)} faces were detected in album.")
    print("")

    return df 
