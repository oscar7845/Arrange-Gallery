import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from utils import file
import os
from pyfiglet import Figlet

# Tested in test_file.py

if __name__ == "__main__":
    f = Figlet(font="slant")
    print(f.renderText("Arrange Gallery"))
    print("----- Handle Duplicates -----")

    album_path = "./images/"

    duplicate_path_tuples = file.find_duplicates(album_path)
    print(f"Found duplicates in album: {len(duplicate_path_tuples)}")
    print(
        f"Uncomment the line #file.remove_duplicates(duplicate_path_tuples) to remove duplicates from album."
    )
    # file.remove_duplicates(duplicate_path_tuples)