"""
Run this file to execute implementation
"""
from utils import ai

if __name__ == "__main__":
    path = "./data/test_images/"
    df = ai.detect_all_faces_in_album(path)
    df = ai.detect_persons(df, tolerance=0.1)
    print(df)
    

