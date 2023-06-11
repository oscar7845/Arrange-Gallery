from utils import ai
from utils import db
import os
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
    db.save(df, store_csv_path)

    # Read df again
    df = db.load(store_csv_path) # Test only, good for caching
    
    print(f"All ids in dataset : {db.get_all_ids(df)}")
    
    print("Rename/Replace id Unknown2 with Person1")
    df = db.replace_id(df,"Unknown2","Person1") # Test only
    
    all_images = db.get_all_images_of_individual(df,"Unknown3") # Test only
    print("Images for id Unknown3: ", all_images)

    print("Merge id Unknown0 and Unknown1.")
    df = db.merge_ids(df, "Unknown0","Unknown1") # Test only

    # Pretty print (DataFrame):
    print(tabulate(df, headers='keys', tablefmt='psql'))



    # TODO: rearrange images
    # TODO: remove duplicates
    # TODO: test on larger dataset

    # TODO: write tests
    # TODO: create test ci/cd
    # TODO: add gui

    # TODO: crop all faces of individual and create large coolage
    # TODO: extra feature: generate gallery slideshow depending on fun settings like color, warmth, peoples or not, resolution, size, format, features, intensity, 