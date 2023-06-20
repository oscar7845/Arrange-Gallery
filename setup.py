from setuptools import setup, find_packages

setup(
    name="ArrangePics",
    version="0.1",
    description="Arrange the pictures in your gallery with face recognition.",
    author="oscar7845",
    url="https://github.com/oscar7845/Arrange-Gallery",
    packages=find_packages(),
    install_requires=[
        "pandas==2.0.1",
        "face_recognition==1.3.0",
        "numpy== 1.24.2",
        "opencv-python==4.7.0.72",
        "tabulate==0.9.0",  # pretty print of dataframe
        "tqdm==4.65.0",  # progress bars
        "black==23.3.0",  # reformatting code
        "pytest==7.3.1",  # testing
        "pyfiglet==0.8.post1",  # pretty print script title
        # Extra features
        "scikit-learn==1.2.2",
        "pytesseract==0.3.10",
        "torch==2.0.1",
        "torchvision==0.15.2",
        "scikit-image==0.20.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
    ],
)