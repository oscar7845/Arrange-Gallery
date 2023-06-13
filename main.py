from utils import file
from utils import run
from utils import db
import os
from tabulate import tabulate
import shutil
from pyfiglet import Figlet

if __name__ == "__main__":
    f = Figlet(font="slant")
    print(f.renderText("text"))

    album_path = "./images/"

    backup_checkpoints = True
    backup_folder = "./data/backups/"

    backup_csv = True
    csv_storage_path = "./data/tmp/fr_db.csv"

    load_df = False
    df_path= "./data/tmp/fr_db.csv"


    if not load_df:
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

        file.save_csv(csv_storage_path, df)

        if backup_csv:
            file.backup(csv_storage_path, backup_folder)

    else: 
        df = db.load_dataframe(df_path)

    # Pretty print (DataFrame):
    # print(tabulate(df, headers='keys', tablefmt='psql'))
    print(df)

    target_path = "./target/"
    if os.path.exists(target_path):
        shutil.rmtree(target_path)

    file.save_all_individual_from_album(target_path,df, allow_copies=False) 

    # TODO: test on larger dataset