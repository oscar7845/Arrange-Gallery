import os
from . import file
from . import ai


def face_recognition_on_album(
    image_path,
    workers=8,
    tolerance=0.6,
    backup_checkpoints=False,
    backup_folder="",
    checkpoint_interval=50,
    checkpoint_path1="./data/tmp/detect_faces_checkpoint.pkl",
    checkpoint_path2="./data/tmp/compare_person_checkpoint.pkl",
):
    df = ai.multi_process_detect_all_faces_in_album(
        image_path,
        workers=workers,
        checkpoint_interval=checkpoint_interval,
        checkpoint_path=checkpoint_path1,
    )
    df = ai.detect_persons(
        df,
        tolerance=tolerance,
        checkpoint_interval=checkpoint_interval
        * 10, 
        checkpoint_path=checkpoint_path2,
    )

    if backup_checkpoints:
        file.backup(checkpoint_path1, backup_folder)
        file.backup(checkpoint_path2, backup_folder)

    try:
        os.remove(checkpoint_path1)
        os.remove(checkpoint_path2)
    except FileNotFoundError as e:
        pass

    return df