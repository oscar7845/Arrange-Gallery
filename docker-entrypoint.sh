#!/bin/bash

if [ "$1" == "--slideshow" ]; then
  echo "Executing the slideshow Python file..."
  python3 /app/features/album_slideshow.py
elif [ "$1" == "--collage" ]; then
  echo "Executing the collage Python file..."
  python3 /app/features/specified_person_collage.py
elif [ "$1" == "--image-resize" ]; then
  echo "Executing the Resize Image Python file..."
  python3 /app/features/resize_images.py
elif [ "$1" == "--remove-duplicates" ]; then
  echo "Executing the Remove Duplicate Images Python file..."
  python3 /app/features/remove_duplicates.py
elif [ "$1" == "--detect-duplicates" ]; then
  echo "Executing the Detect Duplicate Images Python file..."
  python3 /app/features/detect_duplicates.py
elif [ "$1" == "--pytest" ]; then
  echo "Executing pytest..."
  pytest /app/tests
else
  echo "No argument provided, executing the main Python file..."
  python3 /app/main.py
fi
