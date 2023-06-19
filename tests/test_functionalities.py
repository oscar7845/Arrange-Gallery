import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from utils import functionalities
from utils import run
import shutil

# Test majority of dataframe related functions

def test_slideshow_functions():
    album_path = "./images/"
    target_path = "./target/"
    csv_path = "./data/tmp/ss_db.csv"
    checkpoint_interval = 50

    checkpoint_path = "./data/tmp/detect_slideshow_checkpoint.pkl"

    df = functionalities.create_slideshow(
        album_path=album_path,
        target_path=target_path,
        workers=1,
        checkpoint_path=checkpoint_path,
        csv_path=csv_path,
        checkpoint_interval=checkpoint_interval,
    )

    assert len(df) == 7


def test_face_collage_functions():
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

    # feature test
    # Currently just checking that this runs
    functionalities.create_face_collage(df, ["Unknown11"], target_path, (1920, 1080))

    try:
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
    except OSError:
        pass  