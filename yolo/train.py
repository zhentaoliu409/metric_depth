import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import cv2
import yaml
from PIL import Image
from collections import deque
from ultralytics import YOLO
import os

def main():
    # 加载数据集配置
    dataset_path = '../yolo/'
    # Set the path to the YAML file
    yaml_file_path = os.path.join(dataset_path, 'data.yaml')
    # Load and print the contents of the YAML file
    with open(yaml_file_path, 'r') as file:
        yaml_content = yaml.load(file, Loader=yaml.FullLoader)
        print(yaml.dump(yaml_content, default_flow_style=False))

    # Set paths for training and validation image sets
    train_images_path = os.path.join(dataset_path, 'train', 'images')
    valid_images_path = os.path.join(dataset_path, 'valid', 'images')
    test_images_path = os.path.join(dataset_path, 'test', 'images')

    # Initialize counters for the number of images
    num_train_images = 0
    num_valid_images = 0

    # Initialize sets to hold the unique sizes of images
    train_image_sizes = set()
    valid_image_sizes = set()

    # Check train images sizes and count
    for filename in os.listdir(train_images_path):
        if filename.endswith('.jpg'):
            num_train_images += 1
            image_path = os.path.join(train_images_path, filename)
            with Image.open(image_path) as img:
                train_image_sizes.add(img.size)

    # Check validation images sizes and count
    for filename in os.listdir(valid_images_path):
        if filename.endswith('.jpg'):
            num_valid_images += 1
            image_path = os.path.join(valid_images_path, filename)
            with Image.open(image_path) as img:
                valid_image_sizes.add(img.size)

    # Print the results
    print(f"Number of training images: {num_train_images}")
    print(f"Number of validation images: {num_valid_images}")

    # Check if all images in training set have the same size
    if len(train_image_sizes) == 1:
        print(f"All training images have the same size: {train_image_sizes.pop()}")
    else:
        print("Training images have varying sizes.")

    # Check if all images in validation set have the same size
    if len(valid_image_sizes) == 1:
        print(f"All validation images have the same size: {valid_image_sizes.pop()}")
    else:
        print("Validation images have varying sizes.")

    # 分析数据集
    # Set the seed for the random number generator
    random.seed(0)

    # Create a list of image files
    image_files = [f for f in os.listdir(train_images_path) if f.endswith('.jpg')]

    # Randomly select 15 images
    random_images = random.sample(image_files, 15)

    # Create a new figure
    plt.figure(figsize=(19, 12))

    # Loop through each image and display it in a 3x5 grid
    for i, image_file in enumerate(random_images):
        image_path = os.path.join(train_images_path, image_file)
        image = Image.open(image_path)
        plt.subplot(3, 5, i + 1)
        plt.imshow(image)
        plt.axis('off')

    # Add a suptitle
    plt.suptitle('Random Selection of Dataset Images', fontsize=24)

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Deleting unnecessary variable to free up memory
    del image_files

    # 加载模型
    model = YOLO(r"yolo11n-seg.pt", task="segment")  # 使用YOLOv11的分割模型作为起点

    # Train the model on our custom dataset
    results = model.train(
        data=yaml_file_path,     # Path to the dataset configuration file
        epochs=150,              # Number of epochs to train for
        imgsz=640,               # Size of input images as integer
        patience=15,             # Epochs to wait for no observable improvement for early stopping of training
        batch=16,                # Number of images per batch
        optimizer='auto',        # Optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
        lr0=0.0001,              # Initial learning rate
        lrf=0.01,                # Final learning rate (lr0 * lrf)
        dropout=0.25,            # Use dropout regularization
        device=0,                # Device to run on, i.e. cuda device=0
        seed=42                  # Random seed for reproducibility
    )



if __name__ == '__main__':
    main()