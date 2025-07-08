# src/preprocessing.py

import os
import cv2
import numpy as np

def load_data(data_dir, image_size=(128, 128)):
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, image_size)
                images.append(img)
                labels.append(class_name)

    return np.array(images), np.array(labels), class_names
