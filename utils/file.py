from . import db
import os
import shutil
import numpy as np
import hashlib

def find_images(directory):
    image_extensions = [".jpg", ".jpeg", ".png", ".gif"]
    image_paths = []

    for root, _, files in os.walk(directory):
        for file in files:
            if any(extension in file.lower() for extension in image_extensions):
                image_paths.append(os.path.join(root, file))

    return image_paths

def get_appropriate_incremental_name(src_file, dest_folder):
    # Copies a file from src_file to dest_folder, incrementing the file name if a file with the same name already exists.
    file_name = os.path.basename(src_file)
    dest_file = os.path.join(dest_folder, file_name)
    root, ext = os.path.splitext(file_name)
    i = 0
    while os.path.exists(dest_file):
        i += 1
        dest_file = os.path.join(dest_folder, f"{root}_{i}{ext}")
    return dest_file

def save_individual_images(base_path,df,person,ignore_list=[]):

    print(f"Will create folder {person} in {base_path} and copy images there.")
    folder_path = f"{base_path}{person}"
    os.makedirs(folder_path, exist_ok=True)

    if person is None:
        image_list = db.get_all_images_of_non_individuals(df)
    else:
        image_list = db.get_all_images_of_individual(df,person)

    for image_path in image_list:
        if image_path in ignore_list:
            continue
        dest_path = get_appropriate_incremental_name(image_path, folder_path)
        shutil.copy(image_path, dest_path)
        print(f"Copied file to {dest_path}")
       
def save_all_individual_from_album(base_path,df, allow_copies=False):
    persons = db.get_all_ids(df)
    
    print(f"Will now copy all files into individual folders. allow_copies={allow_copies}")
    ignore_list=[]
    
    for i,person in enumerate(persons):
        print(f"Currently on {i+1}/{len(persons)}")

        try:
            if np.isnan(person):
                person = None
        except Exception:
            pass 

        save_individual_images(base_path, df,person, ignore_list)
        if not allow_copies: # make sure images are not copied several times
            ignore_list.extend( db.get_all_images_of_individual(df,person))

def find_duplicates(rootdir):
    hash_dict = {}
    duplicates = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                with open(os.path.join(subdir, file), 'rb') as f:
                    hash = hashlib.md5(f.read()).hexdigest()
                if hash in hash_dict:
                    print(f"Duplicate images found: {os.path.join(subdir, file)} and {hash_dict[hash]}")
                    duplicates.append([os.path.join(subdir, file),hash_dict[hash]])
                else:
                    hash_dict[hash] = os.path.join(subdir, file)

    return duplicates

def remove_duplicates(duplicates):
    for d in duplicates:
        os.remove(d[0])
        print(f"Removed duplicate {d[0]}!")