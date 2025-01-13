#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Liu Zhentao
# @Time    : 2025/1/10 13:23
# Author  : Liu Zhentao
# File    : inference.py
# Licensed under the MIT License
import os
import argparse
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main(model_path, image_path):
    # Define the output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Load the input image using OpenCV
    image = cv2.imread(image_path)

    # Load the model
    model = YOLO(model_path)  # load a custom YOLO model

    # Predict with the model
    results = model(image_path)  # predict on an image

    # Create an empty mask for segmentation
    segmentation_mask = np.zeros_like(image, dtype=np.uint8)
    pothole_label_matrix = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)  # Create an empty label matrix for potholes

    # Iterate over the results
    for i, r in enumerate(results):
        # Iterate through the detected masks
        for j, mask in enumerate(r.masks.xy):
            # Convert the class tensor to an integer
            class_id = int(r.boxes.cls[j].item())  # Extract the class ID as an integer

            # Check if the detected class corresponds to 'pothole' (class ID 0)
            if class_id == 0:
                # Convert mask coordinates to an integer format for drawing
                mask = np.array(mask, dtype=np.int32)

                # Fill the segmentation mask with color (e.g., white for potholes)
                cv2.fillPoly(segmentation_mask, [mask], (0, 255, 0))

                # Fill the pothole label matrix with 1 for detected potholes
                cv2.fillPoly(pothole_label_matrix, [mask], 1)

                # Extract the region of interest (ROI) from the original image
                box = r.boxes.xyxy[j].cpu().numpy().astype(int)
                roi = image[box[1]:box[3], box[0]:box[2]]
                # Save the ROI as a new image
                cv2.imwrite(os.path.join(output_dir, f"segmentation_{i}_{j}.jpg"), roi)
                confidence = r.boxes.conf[j].item()
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(image, f'{confidence:.2f}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Combine the original image with the segmentation mask
    segmentation_result = cv2.addWeighted(image, 1, segmentation_mask, 0.7, 0)

    # Save the output image with segmentation
    cv2.imwrite(os.path.join(output_dir, "output_segmentation.jpg"), segmentation_result)

    # Optionally display the image (make sure you're running in a GUI environment)
    '''
    segmentation_result_rgb = cv2.cvtColor(segmentation_result, cv2.COLOR_BGR2RGB)
    plt.imshow(segmentation_result_rgb)
    plt.title("Segmentation Result")
    plt.axis('off')  # 关闭坐标轴
    plt.show()
    '''

    # Save the pothole label matrix as an image
    cv2.imwrite(os.path.join(output_dir, "pothole_label_matrix.png"), pothole_label_matrix * 255)  # Multiply by 255 to visualize as a binary image

    print(f"Output images saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Inference Script")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the YOLO model")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    main(args.model_path, args.image_path)