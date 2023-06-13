import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from utils import file
from utils import db
from utils import run
import os
import pandas as pd
import shutil


def test_file_save_load():
    csv_storage_path = "./data/tmp/fr_db.csv"
    df = pd.DataFrame(index=range(0), columns=["col1", "col2"])
    file.save_csv(csv_storage_path, df)

    assert os.path.exists(csv_storage_path)

    _df = db.load(csv_storage_path)  
    assert len(df) == len(_df)  


def test_file_handle_functions():
    album_path = "./images/"

    backup_checkpoints = False
    backup_folder = "./data/backups/"

    checkpoint_path1 = "./data/tmp/detect_faces_checkpoint.pkl"
    checkpoint_path2 = "./data/tmp/compare_person_checkpoint.pkl"

    df = run.face_recognition_on_album(
        album_path,
        workers=8,
        tolerance=0.6,
        checkpoint_interval=100,
        backup_checkpoints=backup_checkpoints,
        backup_folder=backup_folder,
        checkpoint_path1=checkpoint_path1,
        checkpoint_path2=checkpoint_path2,
    )

    target_path = "./target/"

    try:
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
    except OSError:
        pass 

    file.save_all_individual_from_album(target_path, df, allow_copies=True)
    assert len(file.find_images(target_path)) == len(df)

    duplicates = file.find_duplicates(target_path)
    assert len(duplicates) == 13

    file.remove_duplicates(duplicates)
    duplicates = file.find_duplicates(target_path)
    assert len(duplicates) == 0

    try:
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
    except OSError:
        pass 

    file.save_all_individual_from_album(target_path, df, allow_copies=False)
    assert len(file.find_images(target_path)) == len(file.find_images(album_path))

    try:
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
    except OSError:
        pass 