import cv2
import os
import numpy as np
from . import db
from . import file

import shutil
import numpy as np
import pickle
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import itertools
from utils import file
import os

from utils import db
from enum import Enum
import face_recognition
import cv2
from sklearn.cluster import KMeans
from skimage.feature import graycomatrix, graycoprops

try:
    import pytesseract
except ImportError:
    print("Tesseract OCR is not installed.")

import torch
import torchvision
import torchvision.transforms as T


class Level(Enum):
    NONE = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3


class Heat(Enum):
    COLD = 0
    WARM = 1


import cv2
import numpy as np


def detect_environment_from_image(image):
    environments = {
        "Outdoor": {"color_hist": [0.1, 0.4, 0.4, 0.1], "glcm": [0.2, 0.2, 0.3, 0.3]},
        "Indoor": {"color_hist": [0.6, 0.3, 0.1, 0.0], "glcm": [0.1, 0.2, 0.4, 0.3]},
        "Natural": {"color_hist": [0.3, 0.2, 0.3, 0.2], "glcm": [0.4, 0.3, 0.2, 0.1]},
        "Urban": {"color_hist": [0.2, 0.5, 0.2, 0.1], "glcm": [0.2, 0.2, 0.4, 0.2]},
        "Industrial": {
            "color_hist": [0.4, 0.3, 0.1, 0.2],
            "glcm": [0.1, 0.3, 0.5, 0.1],
        },
        "Rural": {"color_hist": [0.2, 0.4, 0.3, 0.1], "glcm": [0.3, 0.3, 0.2, 0.2]},
        "Coastal": {"color_hist": [0.1, 0.2, 0.6, 0.1], "glcm": [0.4, 0.1, 0.1, 0.4]}
        # Add more environment categories and their corresponding features
    }

    # Calculate color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [4, 1, 1], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Calculate texture features using the gray image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256))  # Resize the image if needed

    glcm = graycomatrix(
        gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True
    )
    glcm_flat = glcm.flatten()
    glcm = glcm_flat[:4].astype(np.float32)

    # Calculate similarity scores between image features and environment features
    scores = {}

    for environment, features in environments.items():
        color_hist_sim = cv2.compareHist(
            np.array(hist).astype(np.float32),
            np.array(features["color_hist"]).astype(np.float32),
            cv2.HISTCMP_CORREL,
        )
        glcm_sim = cv2.compareHist(
            glcm, np.array(features["glcm"]).astype(np.float32), cv2.HISTCMP_CORREL
        )
        scores[environment] = color_hist_sim + glcm_sim

    max_score = max(scores.values())
    detected_environment = [env for env, score in scores.items() if score == max_score][
        0
    ]

    return detected_environment


def detect_feelings_from_image_pixel_array(pixels):
    avg_color = np.mean(pixels, axis=0)
    emotions = {
        "Angry": [(0, 0, 0), (127, 127, 127)],
        "Happy": [(128, 0, 0), (255, 255, 127)],
        "Sad": [(0, 0, 128), (127, 127, 255)],
        "Neutral": [(0, 128, 128), (255, 255, 255)],
        "Surprised": [(0, 128, 0), (127, 255, 127)],
        "Excited": [(0, 0, 255), (127, 127, 255)],
        "Calm": [(0, 128, 128), (255, 255, 255)],
        "Fearful": [(0, 0, 0), (127, 127, 255)],
        "Disgusted": [(0, 0, 128), (127, 127, 255)],
        "In Love": [(0, 128, 0), (255, 255, 255)],
        "Confused": [(0, 128, 128), (255, 255, 255)],
        "Amused": [(0, 128, 0), (255, 255, 255)],
        "Tired": [(0, 0, 0), (127, 127, 127)],
        "Hopeful": [(0, 128, 0), (127, 255, 127)],
        "Anxious": [(128, 0, 0), (255, 127, 127)],
        "Content": [(0, 128, 128), (127, 255, 255)],
        "Proud": [(0, 0, 128), (127, 127, 255)],
        "Grateful": [(128, 128, 0), (255, 255, 127)],
        "Lonely": [(0, 0, 0), (127, 127, 127)],
        "Surprised": [(128, 0, 128), (255, 127, 255)],
        "Enthusiastic": [(0, 0, 255), (127, 127, 255)],
        "Peaceful": [(128, 128, 0), (255, 255, 127)],
        "Nervous": [(0, 128, 128), (127, 255, 255)],
        "Jealous": [(0, 0, 128), (127, 127, 255)],
        "Excited": [(128, 0, 0), (255, 127, 127)],
        "Curious": [(0, 128, 128), (127, 255, 255)],
        "Hopeless": [(0, 0, 0), (127, 127, 127)],
        "Loving": [(0, 128, 0), (127, 255, 127)],
        "Bored": [(128, 0, 0), (255, 127, 127)],
        "Optimistic": [(0, 128, 128), (127, 255, 255)],
        "Worried": [(0, 0, 128), (127, 127, 255)],
        "Awkward": [(128, 128, 0), (255, 255, 127)],
        "Hurt": [(128, 0, 128), (255, 127, 255)],
        "Relieved": [(0, 128, 255), (127, 255, 255)],
        "Frustrated": [(128, 0, 0), (255, 127, 127)],
        "Cheerful": [(0, 128, 0), (127, 255, 127)],
        "Regretful": [(128, 128, 0), (255, 255, 127)],
        "Apprehensive": [(0, 128, 128), (127, 255, 255)],
        "Eager": [(0, 0, 128), (127, 127, 255)],
        "Gloomy": [(128, 0, 128), (255, 127, 255)],
        "Satisfied": [(0, 128, 128), (127, 255, 255)],
        "Shocked": [(0, 128, 0), (127, 255, 127)],
        "Melancholic": [(128, 0, 0), (255, 127, 127)],
        "Enraged": [(0, 0, 255), (127, 127, 255)],
        "Grumpy": [(128, 128, 0), (255, 255, 127)],
        "Serene": [(0, 0, 128), (127, 127, 255)],
        "Disappointed": [(128, 0, 128), (255, 127, 255)],
        "Astonished": [(0, 128, 128), (127, 255, 255)],
        "Pensive": [(0, 128, 0), (127, 255, 127)],
        "Cautious": [(128, 0, 0), (255, 127, 127)],
        "Excited": [(0, 0, 255), (127, 127, 255)],
        "Thrilled": [(128, 128, 0), (255, 255, 127)],
        "Grateful": [(0, 0, 128), (127, 127, 255)],
        "Inspired": [(128, 0, 128), (255, 127, 255)],
        "Hesitant": [(0, 128, 128), (127, 255, 255)],
        "Despair": [(0, 128, 0), (127, 255, 127)],
        "Determined": [(128, 0, 0), (255, 127, 127)],
        "Fulfilled": [(0, 128, 128), (127, 255, 255)],
        "Perplexed": [(0, 0, 0), (127, 127, 127)],
        "Amazed": [(0, 128, 0), (127, 255, 127)],
        "Skeptical": [(128, 0, 0), (255, 127, 127)],
    }

    image_feelings = []
    for emotion, (lower, upper) in emotions.items():
        if np.all(lower <= avg_color) and np.all(avg_color <= upper):
            image_feelings.append(emotion)

    return image_feelings[:3]


def object_detection_from_image(image, min_score=0.6):
    COCO_OBJECT_CATEGORIES = [
        "__background__",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "N/A",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "N/A",
        "backpack",
        "umbrella",
        "N/A",
        "N/A",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "N/A",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "N/A",
        "dining table",
        "N/A",
        "N/A",
        "toilet",
        "N/A",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "N/A",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
    transform = T.Compose([T.ToTensor()])
    input_image = transform(image)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    model.eval()
    with torch.no_grad():
        predictions = model([input_image])
    detected_objects_labels = predictions[0]["labels"].tolist()
    detected_objects_scores = predictions[0]["scores"].tolist()

    detected_object_names = [
        COCO_OBJECT_CATEGORIES[label] for label in detected_objects_labels
    ]
    detections = list(zip(detected_object_names, detected_objects_scores))

    unique_detections = list(set(detections))
    res = []
    for obj_name, score in unique_detections:
        if score > min_score and obj_name not in res:
            res.append(obj_name)
    return res


def multi_process_slideshow(image_path, debug=False):
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

    if debug:
        print(f"Image: {image_path}")

    # Read image
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)

    # color_dominance
    red_histogram = np.bincount(pixels[:, 0])
    green_histogram = np.bincount(pixels[:, 1])
    blue_histogram = np.bincount(pixels[:, 2])
    dominant_red = np.argmax(red_histogram)
    dominant_green = np.argmax(green_histogram)
    dominant_blue = np.argmax(blue_histogram)
    dominant_color = (dominant_red, dominant_green, dominant_blue)


    if debug:
        print(f"DominantColor: {dominant_color}")

    # color_diversity
    kmeans = KMeans(n_clusters=7, n_init="auto")  # set amount of colors to check
    kmeans.fit(pixels)
    labels = kmeans.labels_
    cluster_counts = np.bincount(labels)
    diversity_color = 1.0 / np.var(cluster_counts)

    if debug:
        print(f"DiversityColor: {diversity_color}")

    # color_warmth
    avg_color = np.mean(pixels, axis=0)
    red, green, blue = avg_color
    color_warmth = (red - blue) / (red + green + blue)

    # image_intensity
    intensity = np.mean(gray_image)
    # image contrast
    contrast = np.std(gray_image)

    if debug:
        print(f"ImageIntensity: {intensity}")
        print(f"ImageContrast: {contrast}")

    # image_quality
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv_image[:, :, 1].mean()
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    quality = saturation * laplacian

    if debug:
        print(f"ImageQuality: {quality}")

    # image_resolution
    height, width, _ = image.shape

    if debug:
        print(f"Resolution: {width}x{height}")

    # image_file_format
    file_extension = os.path.splitext(image_path)[1]
    image_format = file_extension[1:].upper()

    if debug:
        print(f"Format: {image_format}")

    # aspect ratio
    aspect_ratio = width / height

    if debug:
        print(f"AspectRatio: {aspect_ratio}")

    # text
    try:
        text = pytesseract.image_to_string(image)
        if debug:
            print(f"Text: {text}")
    except Exception:
        text = None

    # image smooth edges
    gradient = cv2.Laplacian(image, cv2.CV_64F)
    gradient_magnitude = np.abs(gradient).mean()

    if debug:
        print(f"SmoothEdges: {gradient_magnitude}")

    top_three_feelings = detect_feelings_from_image_pixel_array(pixels)

    if debug:
        print(f"Feelings: {top_three_feelings}")

    # environment
    environment = detect_environment_from_image(image)

    if debug:
        print(f"Environement: {environment}")

    # sift
    sift = cv2.SIFT_create()
    keypoints, __descriptors = sift.detectAndCompute(gray_image, None)

    if debug:
        print(f"Sift: {len(keypoints)}")

    # detected_objects
    detected_objects_labels = object_detection_from_image(image)

    if debug:
        print(f"Detection: {detected_objects_labels}")

    # Face Detection
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image, model="hog")

    if debug:
        print(f"Persons: {len(face_locations)}")
        print("Done!")

    new_row = pd.DataFrame(
        {
            "image_path": [image_path],
            "color_dominance": [dominant_color],
            "color_diversity": [diversity_color],
            "color_warmth": [color_warmth],
            "image_intensity": [intensity],
            "image_contrast": [contrast],
            "image_quality": [quality],
            "image_resolution": [[width, height]],
            "image_file_format": [image_format],
            "aspect_ratio": [aspect_ratio],
            "text": [text],
            "image_smooth_edges": [gradient_magnitude],
            "image_feeling": [[top_three_feelings]],
            "environment": [environment],
            "sift_features": [len(keypoints)],
            "people": [len(face_locations)],
            "objects": [[detected_objects_labels]],
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
    # Load checkpoint if exist
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
    csv_path="./data/tmp/ss_db.csv",
    checkpoint_interval=50,

    # Color analysis
    color_dominance=None,  # bgr, check for color dominant occurrences
    color_diversity=None,  # high large diversity
    color_warmth=None,  # low = cold, high = warm
    image_intensity=None,
    image_contrast=None,

    # Image file specific
    min_image_quality=None,
    min_image_resolution=(0, 0),
    max_image_resolution=(4000, 4000),
    image_file_formats=None,
    aspect_ratio_range=None,

    # Text detection
    text_amount=None,
    text=None,

    # Corner detection
    image_smooth_edges=None,

    # Sentient Analysis
    image_feeling=None,

    # Environment Analysis
    environment=None,  # "outside"

    # Feature extraction
    sift_features=None,

    # Face detection
    people=None,

    # Object detection
    allowed_objects=None,  # None, allows all
    not_allowed_objects=None,  # None, ignores None
):
    df = generate_slideshow_dataframe(
        album_path,
        workers=workers,
        checkpoint_interval=checkpoint_interval,
        checkpoint_path=checkpoint_path,
    )
    pd.set_option('display.max_columns', None)
    print(df)

    try:
        os.remove(checkpoint_path)
    except FileNotFoundError as e:
        pass

    # Save df to file
    file.save_csv(csv_path, df)

    create_slideshow_from_df_and_filters(
        df,
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
        text_amount,
        text,
        image_smooth_edges,
        image_feeling,
        environment,
        sift_features,
        people,
        allowed_objects,
        not_allowed_objects,
    )
    return df


import math


def color_distance(color1, color2):
    r_diff = color1[0] - color2[0]
    g_diff = color1[1] - color2[1]
    b_diff = color1[2] - color2[2]
    return math.sqrt(r_diff**2 + g_diff**2 + b_diff**2)


def create_slideshow_from_df_and_filters(
    df,
    target_path,
    color_dominance=None,  # bgr, check for color dominant occurrences
    color_diversity=None,  # high large diversity
    color_warmth=None,  # low = cold, high = warm
    image_intensity=None,
    image_contrast=None,

    # Image file specific
    min_image_quality=None,
    min_image_resolution=(0, 0),
    max_image_resolution=(4000, 4000),
    image_file_formats=None,
    aspect_ratio_range=(0, 4),

    # Text detection
    text_amount=None,
    text=None,

    # Corner detection
    image_smooth_edges=None,

    # Sentient Analysis
    image_feeling=None,

    # Environment Analysis
    environment=None,  # "outside"

    # Feature extraction
    sift_features=None,

    # Face detection
    people=None,

    # Object detection
    allowed_objects=None,  # None, allows all
    not_allowed_objects=None,  # None, ignores None
):
    print("----- Creating Slideshow -----")

    # Create target folder
    dest = file.get_appropriate_incremental_name("slideshow", target_path)
    os.makedirs(dest, exist_ok=True)

    slide_show_image_paths = []
    for _, row in df.iterrows():
        add = True

        if color_dominance is not None:
            if color_distance(row["color_dominance"], color_dominance) > 100:
                continue

        if color_diversity is not None:
            if color_diversity == Level.LOW:
                print(row["color_diversity"])
                if row["color_diversity"] >= 0.00000002:
                    continue
            elif color_diversity == Level.MODERATE:
                if row["color_diversity"] < 0.00000002 or row["color_diversity"] > 0.000005:
                    continue
            elif color_diversity == Level.HIGH:
                if row["color_diversity"] <= 0.000005:
                    continue

        if color_warmth is not None:
            if color_warmth == Heat.COLD:
                if row["color_warmth"] > 0.025:
                    continue
            elif color_warmth == Heat.WARM:
                if row["color_warmth"] < 0.025:
                    continue

        if image_intensity is not None:
            if image_intensity == Level.LOW:
                if row["image_intensity"] >= 50:
                    continue
            elif image_intensity == Level.MODERATE:
                if row["image_intensity"] < 50 or row["image_intensity"] > 150:
                    continue
            elif image_intensity == Level.HIGH:
                if row["image_intensity"] <= 150:
                    continue

        if image_contrast is not None:
            if image_contrast == Level.LOW:
                if row["image_contrast"] >= 40:
                    continue
            elif image_contrast == Level.MODERATE:
                if row["image_contrast"] < 40 or row["image_contrast"] > 60:
                    continue
            elif image_contrast == Level.HIGH:
                if row["image_contrast"] <= 60:
                    continue

        if min_image_quality is not None:
            if min_image_quality == Level.LOW:
                if row["image_quality"] >= 35000:
                    continue
            elif min_image_quality == Level.MODERATE:
                if row["image_quality"] < 35000 or row["image_quality"] > 150000:
                    continue
            elif min_image_quality == Level.HIGH:
                if row["image_quality"] <= 150000:
                    continue

        if min_image_resolution is not None:
            if (
                min_image_resolution[0] > row["image_resolution"][0]
                or min_image_resolution[1] > row["image_resolution"][1]
            ):
                continue

        if max_image_resolution is not None:
            if (
                max_image_resolution[0] < row["image_resolution"][0]
                or max_image_resolution[1] < row["image_resolution"][1]
            ):
                continue

        if image_file_formats is not None:
            if row["image_file_format"] not in image_file_formats:
                continue

        if aspect_ratio_range is not None:
            if (
                aspect_ratio_range[0]
                < row["aspect_ratio_range"]
                < aspect_ratio_range[1]
            ):
                continue

        if text_amount is not None:
            if text_amount > len(row["text"]):
                continue

        if text is not None:
            for word in text:
                if word not in row["text"]:
                    continue

        if image_smooth_edges is not None:
            if image_smooth_edges == Level.LOW:
                if row["image_smooth_edges"] >= 0.2:
                    continue
            elif image_smooth_edges == Level.MODERATE:
                if row["image_smooth_edges"] < 0.2 or row["image_smooth_edges"] > 0.5:
                    continue
            elif image_smooth_edges == Level.HIGH:
                if row["image_smooth_edges"] <= 0.5:
                    continue

        if image_feeling is not None:
            for feel in image_feeling:
                if feel not in row["image_feeling"]:
                    continue

        if environment is not None:
            if row["environment"] not in environment:
                continue

        if sift_features is not None:
            if sift_features == Level.LOW:
                if row["sift_features"] >= 100:
                    continue
            elif sift_features == Level.MODERATE:
                if row["sift_features"] < 100 or row["sift_features"] > 200:
                    continue
            elif sift_features == Level.HIGH:
                if row["sift_features"] <= 200:
                    continue

        if people is not None:
            if people == Level.NONE:
                if row["people"] != 0:
                    continue
            elif people == Level.LOW:
                if row["people"] >= 2:
                    continue
            elif people == Level.MODERATE:
                if row["people"] < 2 or row["people"] > 5:
                    continue
            elif people == Level.HIGH:
                if row["people"] <= 5:
                    continue

        if allowed_objects is not None:
            skip=False
            for obj in allowed_objects:
                if obj.lower() not in row["objects"][0]:
                    skip=True
                    break
            if skip:
                continue

        if not_allowed_objects is not None:
            skip=False
            for obj in not_allowed_objects:
                if obj.lower() in row["objects"][0]:
                    skip=True
                    break
            if skip:
                continue

        slide_show_image_paths.append(row["image_path"])

    print(f"Amount of images in slideshow: {len(slide_show_image_paths)}")

    for image in slide_show_image_paths:
        dest_path = file.get_appropriate_incremental_name(image, dest)
        shutil.copy(image, dest_path)
        print(f"Copied file to {dest_path}")


def create_face_collage(df, persons, target_path, resolution):
    print("----- Generate Collage -----")
    cropped_faces = []
    for person in persons:
        personal_df = db.get_all_occurrences_of_individual(df, person)

        # collect cropped faces
        for _, row in personal_df.iterrows():
            img = cv2.imread(row["image_path"])
            top, right, bottom, left = row["box"]
            width = right - left
            height = bottom - top
            x = left
            y = top
            cropped_faces.append(img[y : y + height, x : x + width])

    merged_image = merge_images(cropped_faces, resolution[0], resolution[1])
    if merged_image is None:
        print(f"No faces found for person {person}.")
        return None

    os.makedirs(target_path, exist_ok=True)
    dest = file.get_appropriate_incremental_name("face_collage.png", target_path)
    cv2.imwrite(dest, merged_image)
    print(f"Saved collage at {dest}.")

    return merged_image


def merge_images(
    images, output_width, output_height
):
    if len(images) == 0:
        return None

    num_images = len(images)
    num_cols = int(np.ceil(np.sqrt(num_images)))
    num_rows = int(np.ceil(num_images / num_cols))

    print(f"Amount of images to merge: {len(images)}")
    print(f"Image merge dimensions: {num_cols}x{num_rows}")
    print(f"Output image resolutiong: {output_width}x{output_height}")

    subimage_width = int(output_width / num_cols)
    subimage_height = int(output_height / num_rows)

    output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        row_idx = i // num_cols
        col_idx = i % num_cols

        img_resized = cv2.resize(img, (subimage_width, subimage_height))

        x = col_idx * subimage_width
        y = row_idx * subimage_height

        output_image[y : y + subimage_height, x : x + subimage_width, :] = img_resized

    return output_image