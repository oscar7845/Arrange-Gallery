import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import shutil
from utils import db
from utils import file
from utils import run
from utils import functionalities
import os
from pyfiglet import Figlet


if __name__ == "__main__":
    f = Figlet(font="slant")
    print(f.renderText("Arrange Gallery"))

    album_path = "./images/"
    target_path = "./target/"

    # Album root dir
    album_path = "./images/"

    # Backup checkpoints to use them after to remove steps from calculations, like caching
    backup_checkpoints = True
    backup_folder = "./data/backups/"

    # Backup Dataframe to avoid it being accidentally replaced during next run
    backup_csv = True
    csv_storage_path = "./data/tmp/fr_db.csv"

    # Load Dataframe instead of performing calculation
    load_df = False
    df_path = "./data/tmp/fr_db.csv"

    #-------------------------------#
    # Storage paths for checkpoints #
    #-------------------------------#

    if not load_df:
        checkpoint_path1 = "./data/tmp/detect_faces_checkpoint.pkl"
        checkpoint_path2 = "./data/tmp/compare_person_checkpoint.pkl"

        # Do calculations with checkpointing to create dataset of all images and faces recognized and compared to personal ids
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

        # Save to CSV file
        file.save_csv(csv_storage_path, df)

        if backup_csv:
            file.backup(csv_storage_path, backup_folder)

    else:
        df = db.load_dataframe(df_path)

    # Pretty print (DataFrame):
    # print(tabulate(df, headers='keys', tablefmt='psql'))
    print(df)

    collage = functionalities.create_face_collage(df, ["Unknown11"], target_path, (1920, 1080))