# Pyrmont Brewery Raspbeery Pi Yeast Viability Counter
# Kevin Staunton-Lambert
# Copyright (c) Pyrmont Brewery 2007-2024

import os
import cv2
import numpy as np

train_path_pos = "trainingset/pos"
train_path_neg = "trainingset/neg"

# Load yeast viability training set images
pos_images = []
for filename in os.listdir(train_path_pos):
    pos_images.append(cv2.imread(os.path.join(train_path_pos, filename)))

neg_images = []
for filename in os.listdir(train_path_neg):
    neg_images.append(cv2.imread(os.path.join(train_path_neg, filename)))

# Drop the colour
gray_pos_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in pos_images]
gray_neg_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in neg_images]

# TODO: annotate and verify

