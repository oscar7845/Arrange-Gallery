import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from utils import db
from utils import run

# Test majority of dataframe related functions

def test_db_functions():
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

    assert len(db.get_all_ids(df)) == 15
    print(f"Amount of unique ids in dataset : {len(db.get_all_ids(df))}")

    print("Rename/Replace id Unknown2 with Person1")
    assert len(db.get_all_occurrences_of_individual(df, "Unknown2")) == 1
    df = db.replace_id(df, "Unknown2", "Person1")  # Test only
    assert len(db.get_all_occurrences_of_individual(df, "Person1")) == 1
    assert len(db.get_all_occurrences_of_individual(df, "Unknown2")) == 0

    all_images = db.get_all_images_of_individual(df, "Unknown3")
    print("Images for id Unknown3: ", all_images)
    assert len(all_images) == 1