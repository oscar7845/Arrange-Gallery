import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.functionalities import Level
from utils.functionalities import Heat
from pyfiglet import Figlet
from utils import functionalities
import shutil

if __name__ == "__main__":
    f = Figlet(font="slant")
    print(f.renderText("Arrange Gallery"))

    album_path = "./images/"
    target_path = "./target/"

    #-------------------#
    # Slideshow filters #
    #-------------------#
    
    # Are ignored if set to None

    # Color analysis
    # bgr, check for color dominant occurrences, uses color euclidean distance > 100
    color_dominance = None  # example (0,0,0)

    # high value -> large diversity
    color_diversity = None  # example Level.MODERATE

    # Set heat color of images
    color_warmth = None  # example Heat.COLD

    # Image file specific
    # Intensity of image values
    image_intensity = None  # example Level.MODERATE

    # Contrast of images
    image_contrast = None  # example Level.MODERATE

    # Set image quality, uses saturn and laplacian
    min_image_quality = None  # example Level.LOW

    # Set minimum resolution
    min_image_resolution = None  # example (0, 0)

    # Set maximum resolution
    max_image_resolution = None  # example (4000, 4000)

    # Image file format
    # List of formats, strings should be UPPER
    image_file_formats = None  # example ["JPG"]

    # Image aspect ratio range
    aspect_ratio_range = None  # example (0, 2)

    #----------------#
    # Text detection #
    #----------------#

    # Calculates the amount of words found in images
    text_amount = None  # example 1

    # Search for words in images
    text = None  # example ["text"]

    # Corner detection
    image_smooth_edges = None  # example Level.MODERATE

    # Sentient Analysis - detects feelings from image average color
    image_feeling = None  # example ["Calm"] - list of feelings

    # Environment Analysis - estimate environmet from glcm
    environment = None  # example ["Urban"] - list of environments

    # Feature extraction - SIFT feature extraction
    sift_features = None  # example Level.LOW

    # Face detection - amount of people in images - dlib face detection
    people = None  # example, Level.LOW # level

    # Object detection - RCNN using COCO dataset
    allowed_objects = None  # example ["tie"] create a list of strings of object names
    not_allowed_objects = None  # example ["remote"]

    workers = 1  # amount of concurrent processes
    checkpoint_path = "./data/tmp/slideshow_checkpoint.pkl"
    csv_path = "./data/tmp/ss_db.csv"
    checkpoint_interval = 50

    # Remove current target
    try:
        target_path = "./target/"
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
    except OSError:
        pass 

    functionalities.create_slideshow(
        album_path=album_path,
        target_path=target_path,
        workers=workers,
        checkpoint_path=checkpoint_path,
        csv_path=csv_path,
        checkpoint_interval=checkpoint_interval,
        color_dominance=color_dominance,
        color_diversity=color_diversity,
        color_warmth=color_warmth,
        image_intensity=image_intensity,
        image_contrast=image_contrast,
        min_image_quality=min_image_quality,
        min_image_resolution=min_image_resolution,
        max_image_resolution=max_image_resolution,
        image_file_formats=image_file_formats,
        aspect_ratio_range=aspect_ratio_range,
        text_amount=text_amount,
        text=text,
        image_smooth_edges=image_smooth_edges,
        image_feeling=image_feeling,
        environment=environment,
        sift_features=sift_features,
        people=people,
        allowed_objects=allowed_objects,
        not_allowed_objects=not_allowed_objects,
    )