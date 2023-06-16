import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pickle
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import itertools
from utils import file
import os
from pyfiglet import Figlet
from utils import db
from enum import Enum
import face_recognition
import cv2
from sklearn.cluster import KMeans
import pytesseract
import torch
import torchvision
import torchvision.transforms as T

class Level(Enum):
    NONE = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3

import cv2
import numpy as np

def detect_environment_from_image(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    gray = cv2.resize(gray, (256, 256))  # Resize the image if needed
    glcm = np.zeros((256, 256))
    for i in range(256):
        for j in range(256):
            if j < 255:
                glcm[gray[i][j]][gray[i][j+1]] += 1
    glcm = glcm / np.sum(glcm)

    environments = {
        "Outdoor": {
            "color_hist": [0.1, 0.4, 0.4, 0.1],
            "glcm": [0.2, 0.2, 0.3, 0.3]
        },
        "Indoor": {
            "color_hist": [0.6, 0.3, 0.1, 0.0],
            "glcm": [0.1, 0.2, 0.4, 0.3]
        },
        "Natural": {
            "color_hist": [0.3, 0.2, 0.3, 0.2],
            "glcm": [0.4, 0.3, 0.2, 0.1]
        },
        "Urban": {
            "color_hist": [0.2, 0.5, 0.2, 0.1],
            "glcm": [0.2, 0.2, 0.4, 0.2]
        },
        "Industrial": {
            "color_hist": [0.4, 0.3, 0.1, 0.2],
            "glcm": [0.1, 0.3, 0.5, 0.1]
        },
        "Rural": {
            "color_hist": [0.2, 0.4, 0.3, 0.1],
            "glcm": [0.3, 0.3, 0.2, 0.2]
        },
        "Coastal": {
            "color_hist": [0.1, 0.2, 0.6, 0.1],
            "glcm": [0.4, 0.1, 0.1, 0.4]
        }
    }

    scores = {}
    for environment, features in environments.items():
        color_hist_sim = cv2.compareHist(hist, np.array(features["color_hist"]), cv2.HISTCMP_CORREL)
        glcm_sim = cv2.compareHist(glcm, np.array(features["glcm"]), cv2.HISTCMP_CORREL)
        scores[environment] = color_hist_sim + glcm_sim

    return max(scores, key=scores.get)

def multi_process_slideshow(image_path):
    df = db.create(
        [
            "image_path",
            "color_dominance",
            "color_diversity",
            "color_warmth",
            "image_intensity",
            "image_contrast",
            "image_quality",
            "image_resolution",
            "image_file_format",
            "aspect_ratio_range",
            "text",
            "image_smooth_edges",
            "image_feeling",
            "environment",
            "sift_features",
            "people",
            "objects",
           
        ]
    )
    
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)

    histogram = np.bincount(pixels[:, 0])
    dominant_color = np.argmax(histogram)

    kmeans = KMeans(n_clusters=7) # set amount of colors to check
    kmeans.fit(pixels)
    labels = kmeans.labels_
    cluster_counts = np.bincount(labels)
    diversity_color = 1.0 / np.var(cluster_counts)

    avg_color = np.mean(pixels, axis=0)
    red, green, blue = avg_color
    color_warmth = (red - blue) / (red + green + blue)

    intensity = np.mean(gray_image)

    contrast = np.std(gray_image)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv_image[:, :, 1].mean()
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    quality = saturation * laplacian

    height, width, _ = image.shape

    file_extension = os.path.splitext(image_path)[1]
    image_format = file_extension[1:].upper()
   
    aspect_ratio = width / height

    text = pytesseract.image_to_string(image)
    
    gradient = cv2.Laplacian(image, cv2.CV_64F)
    gradient_magnitude = np.abs(gradient).mean()

    avg_color = np.mean(pixels, axis=0)
    emotions = {
        "Angry": [(220, 20, 60), (255, 99, 71)],
        "Happy": [(255, 215, 0), (255, 255, 0)],
        "Sad": [(70, 130, 180), (135, 206, 250)],
        "Neutral": [(192, 192, 192), (211, 211, 211)],
        "Surprised": [(0, 128, 0), (50, 205, 50)],
        "Excited": [(255, 0, 0), (255, 99, 71)],
        "Calm": [(135, 206, 250), (176, 224, 230)],
        "Fearful": [(0, 0, 0), (139, 0, 139)],
        "Disgusted": [(128, 128, 0), (173, 255, 47)],
        "In Love": [(255, 105, 180), (255, 20, 147)],
        "Confused": [(0, 128, 128), (0, 139, 139)],
        "Amused": [(255, 165, 0), (255, 215, 0)],
        "Tired": [(160, 82, 45), (139, 69, 19)]
    }

    image_feeling = None
    for emotion, (lower, upper) in emotions.items():
        if np.all(lower <= avg_color) and np.all(avg_color <= upper):
            image_feeling = emotion

    environment = detect_environment_from_image(image)

    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    transform = T.Compose([T.ToTensor()])
    input_image = transform(image)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    with torch.no_grad():
        predictions = model([input_image])
    detected_objects_labels = predictions[0]["labels"]

    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image, model="hog")

    new_row = pd.DataFrame(
        {
            "image_path":  [image_path],
            "color_dominance": [dominant_color],
            "color_diversity": [diversity_color],
            "color_warmth": [color_warmth],
            "image_intensity": [intensity],
            "image_contrast":[contrast],
            "image_quality": [quality],
            "image_resolution": [width,height],
            "image_file_format": [image_format],
            "aspect_ratio": [aspect_ratio],
            "text": [text],
            "image_smooth_edges": [gradient_magnitude],
            "image_feeling": [image_feeling],
            "environment": [environment],
            "sift_features": [len(keypoints)],
            "people": [len(face_locations)],
            "objects": [detected_objects_labels]
           
        }
    )
    df = pd.concat([df, new_row], ignore_index=True)

    return df


def generate_slideshow_dataframe(
    path,
    workers=8,
    checkpoint_path="./data/tmp/slideshow_checkpoint.pkl",
    checkpoint_interval=50,
):
    image_paths = file.find_images(path)

    if len(image_paths) < checkpoint_interval:
        splitted_paths = [image_paths]
    else:
        splitted_paths = np.array_split(
            image_paths, len(image_paths) / checkpoint_interval
        )

    print("----- Slideshow generation -----")
    print(f"Number of images: {len(image_paths)}")
    print(f"Number of batches: {int(len(image_paths)/checkpoint_interval)+1}")
    print(f"Number of checkpoint_interval: {checkpoint_interval}")

    i = 0
    dfss = []
    try:
        with open(checkpoint_path, "rb") as f:
            dfss, i = pickle.load(f)
            print(f"Checkpoint found, continuing from id {i}")
    except FileNotFoundError:
        pass

    with Pool(workers) as pool:
        for j, path in enumerate(
            tqdm(itertools.islice(splitted_paths, i, None), total=len(splitted_paths)),
            start=i,
        ):
            dfs = []
            for result in tqdm(
                pool.imap(multi_process_slideshow, path), total=len(path)
            ):
                dfs.append(result)

            try:
                dfss.append(pd.concat(dfs, ignore_index=True))
            except ValueError as ve:
                pass  

            print()
            print()
            print(f"  Checkpoint - {j+1}/{len(splitted_paths)}")
            print()

            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            with open(checkpoint_path, "wb+") as f:
                pickle.dump((dfss, j + 1), f)

        pool.close()
        pool.join()

    try:
        df = pd.concat(dfss, ignore_index=True)
    except ValueError as ve:
        pass 

    print("")
    print(" ---- Statistics -----")
    print(f"A total of {len(df)} slideshow.")
    print("")

    return df


def create_slideshow(
    album_path,
    target_path,
    workers=8,
    checkpoint_path="./data/tmp/slideshow_checkpoint.pkl",
    checkpoint_interval=50,

    color_dominance=(0, 0, 0),  # bgr, check for color dominant occurrences
    color_diversity=Level.LOW,  # high large diversity
    color_warmth=0,  # low = cold, high = warm
    image_intensity=Level.LOW,
    image_contrast=Level.LOW,

    min_image_quality=Level.LOW,
    min_image_resolution=(0, 0),
    max_image_resolution=(4000, 4000),
    image_file_formats=[".jpg", ".png", ".jpeg", ".gif"],
    aspect_ratio_range=(0, 2),

    text=Level.LOW,

    image_smooth_edges=Level.LOW,

    image_feeling="calm",

    environment="inside",  # "outside"

    sift_features=Level.LOW,

    people=Level.NONE,

    allowed_objects=None,  # None, allows all
    not_allowed_objects=None,  # None, ignores None
):
    df = generate_slideshow_dataframe(
        album_path,
    )

if __name__ == "__main__":
    f = Figlet(font="slant")
    print(f.renderText("text"))

    album_path = "./data/images/"
    target_path = "./target/"


    color_dominance = (0, 0, 0)  # bgr, check for color dominant occurrences
    color_diversity = Level.LOW  # high large diversity
    color_warmth = 0  # low = cold, high = warm
    image_intensity=Level.LOW
    image_contrast=Level.LOW

    min_image_quality = Level.LOW
    min_image_resolution = (0, 0)
    max_image_resolution = (4000, 4000)
    image_file_formats = [".jpg", ".png", ".jpeg", ".gif"]
    aspect_ratio_range = (0, 2)

    text = Level.LOW
    
    image_smooth_edges = Level.LOW

    image_feeling = "calm"

    environment = "inside"  # "outside"

    sift_features = Level.LOW

    people = Level.NONE

    allowed_objects = None  # create a list of strings of object names
    not_allowed_objects = None

    create_slideshow(
        album_path,
        target_path,
        color_dominance,
        color_diversity,
        color_warmth,
        image_intensity,
        image_contrast,
        min_image_quality,
        min_image_resolution,
        max_image_resolution,
        image_file_formats,
        aspect_ratio_range,
        text,
        image_smooth_edges,
        image_feeling,
        environment,
        sift_features,
        people,
        allowed_objects,
        not_allowed_objects,
    )