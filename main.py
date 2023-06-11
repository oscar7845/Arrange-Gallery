from utils import ai
import os
import pandas as pd
from tabulate import tabulate

if __name__ == "__main__":
    path = "./images/"
    df = ai.detect_all_faces_in_album(path, multi_process=True, workers=8, rescale=None, show_images=False, images_show_timer=1000, estimate_time=True)
    df = ai.detect_persons(df, tolerance=0.6)

    # Save to CSV file safe
    store_csv_path = './data/tmp/tmp.csv'
    dir_path = os.path.dirname(store_csv_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    df.to_csv(store_csv_path, index=False)

    # Read df again
    df = pd.read_csv(store_csv_path)

    # Pretty print (DataFrame):
    print(tabulate(df, headers='keys', tablefmt='psql'))

    # TODO: get all ids
    # TODO: replace name of individual
    # TODO: get all images of individual
    # TODO: merge persons
    # TODO: rearrange images
    # TODO: remove duplicates
    # TODO: create larger test dataset

    # TODO: add gui
    # TODO: write tests
    # TODO: create test ci/cd
    