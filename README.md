# AI-Based Road Damage Detection and Severity Analysis

A computer vision project for detecting road damage, identifying damage types, and analyzing severity using YOLO.

The project is currently focused on dataset validation, duplicate detection, and preparing a leakage-safe train/validation/test split before model training.

## Project Goals

- Detect road damage using bounding boxes
- Identify different road damage types
- Distinguish normal and severe damage variants
- Display prediction confidence
- Provide a simple severity result
- Later support image and video inference
- Later build a Streamlit interface

## Damage Classes

The dataset contains 8 classes:

1. Longitudinal Crack
2. Longitudinal Crack Wide
3. Transverse Crack
4. Transverse Crack Wide
5. Alligator Crack
6. Alligator Crack Sunken
7. Pothole
8. Pothole Deep

## Dataset Summary

- Total images: 25,600
- Total bounding-box annotations: 35,910
- Background images: 8,318
- Corrupt images found: 0
- Invalid annotation lines found: 0

The dataset contains:

- 12,800 original images
- 12,800 horizontally mirrored images

## Work Completed

So far, the following checks have been completed:

- Image and label file consistency
- YOLO annotation validation
- Corrupt image checking
- Background image analysis
- Image resolution analysis
- Filename analysis
- Class distribution analysis
- Original and mirrored image pairing
- Mirrored annotation verification
- Exact duplicate checking
- Near-duplicate analysis using pHash
- ORB-based visual similarity comparison
- Manual verification of same-scene image pairs

## Current Status

The project is currently at the dataset preparation stage.

The next step is to create leakage-safe image groups so that:

- Original and mirrored images stay together
- Same-scene duplicate images stay together
- Related images do not appear across train, validation, and test sets

## Next Steps

1. Build final leakage-safe groups
2. Create train/validation/test splits
3. Check class balance across splits
4. Prepare YOLO dataset structure
5. Train a YOLO model
6. Evaluate model performance
7. Add image/video inference
8. Build a Streamlit interface

## Tech Stack

- Python
- OpenCV
- Pillow
- ImageHash
- Ultralytics YOLO (planned)
- Streamlit (planned)

## Status

Work in progress.