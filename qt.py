#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Liu Zhentao
# @Time    : 2025/1/10 13:23
# Author  : Liu Zhentao
# File    : qt.py
# Licensed under the MIT License
import sys
import torch
import cv2
import numpy as np
import subprocess
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class Worker(QThread):
    yolo_finished = pyqtSignal(str)
    depth_finished = pyqtSignal(str, str)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):

        torch.cuda.empty_cache()
        # Run YOLO inference
        subprocess.run(["python", "yolo/inference.py", "--model-path", "yolo/runs/segment/train7/weights/best.pt", "--image-path", self.image_path])

        # Path to the YOLO output image
        yolo_output_dir = "./yolo/outputs"
        segmentation_image_path = f"{yolo_output_dir}/output_segmentation.jpg"
        self.yolo_finished.emit(segmentation_image_path)

        # Run depth estimation and point cloud generation using the original image
        output_dir = "./outputs"
        depth_map_path = f"{output_dir}/{self.image_path.split('/')[-1].split('.')[0]}.png"
        ply_file = f"{output_dir}/{self.image_path.split('/')[-1].split('.')[0]}.ply"

        subprocess.run(["python", "run.py", "--encoder", "vitl", "--load-from",
                        "checkpoints/depth_anything_v2_metric_vkitti_vitl.pth", "--max-depth", "80", "--pred-only", "--img-path",
                        self.image_path, "--outdir", output_dir])
        subprocess.run(["python", "depth_to_pointcloud.py", "--encoder", "vitl", "--load-from",
                        "checkpoints/depth_anything_v2_metric_vkitti_vitl.pth", "--max-depth", "20", "--img-path",
                        self.image_path, "--outdir", output_dir])

        self.depth_finished.emit(depth_map_path, ply_file)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionRepair: Pothole Detection and 3D Road Restoration")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.image_label = QLabel(self)
        self.segmentation_label = QLabel(self)
        self.depth_map_label = QLabel(self)
        self.push_button_select_image = QPushButton("Select Image", self)
        self.push_button_fill_plane = QPushButton("Fill With Plane", self)
        self.push_button_fill_mesh = QPushButton("Fill With Mesh", self)
        self.push_button_fill_curvature = QPushButton("Fill With Curvature", self)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.push_button_select_image)
        top_layout.addWidget(self.push_button_fill_plane)
        top_layout.addWidget(self.push_button_fill_mesh)
        top_layout.addWidget(self.push_button_fill_curvature)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.image_label)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.segmentation_label)
        right_layout.addWidget(self.depth_map_label)

        bottom_layout = QHBoxLayout()
        bottom_layout.addLayout(left_layout)
        bottom_layout.addLayout(right_layout)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(bottom_layout)

        self.central_widget.setLayout(main_layout)

        self.push_button_select_image.clicked.connect(self.on_select_image_clicked)
        self.push_button_fill_plane.clicked.connect(self.on_fill_plane_clicked)
        self.push_button_fill_mesh.clicked.connect(self.on_fill_mesh_clicked)
        self.push_button_fill_curvature.clicked.connect(self.on_fill_curvature_clicked)

        self.ply_file = None

    def on_select_image_clicked(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.bmp *.jpeg)")
        if file_name:
            self.display_image(file_name)
            self.segmentation_label.clear()
            self.depth_map_label.clear()
            self.worker = Worker(file_name)
            self.worker.yolo_finished.connect(self.on_yolo_finished)
            self.worker.depth_finished.connect(self.on_depth_finished)
            self.worker.start()

    def display_image(self, image_path):
        image = QImage(image_path)
        self.image_label.setPixmap(QPixmap.fromImage(image).scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def on_yolo_finished(self, segmentation_image_path):
        self.display_segmentation(segmentation_image_path)

    def on_depth_finished(self, depth_map_path, ply_file):
        self.display_depth_map(depth_map_path)
        self.ply_file = ply_file

    def display_segmentation(self, segmentation_image_path):
        segmentation_image = QImage(segmentation_image_path)
        self.segmentation_label.setPixmap(QPixmap.fromImage(segmentation_image).scaled(self.segmentation_label.size(), Qt.KeepAspectRatio))

    def display_depth_map(self, depth_map_path):
        depth_map = cv2.imread(depth_map_path, cv2.IMREAD_COLOR)
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2RGB)
        depth_map_image = QImage(depth_map.data, depth_map.shape[1], depth_map.shape[0], depth_map.strides[0], QImage.Format_RGB888)
        self.depth_map_label.setPixmap(QPixmap.fromImage(depth_map_image).scaled(self.depth_map_label.size(), Qt.KeepAspectRatio))

    def on_fill_plane_clicked(self):
        if self.ply_file:
            subprocess.run(["python", "fill_plane.py", "--file", self.ply_file])
        else:
            print("Please import an image and wait for depth estimation and point cloud conversion.")

    def on_fill_mesh_clicked(self):
        if self.ply_file:
            subprocess.run(["python", "fill_mesh.py", "--file", self.ply_file])
        else:
            print("Please import an image and wait for depth estimation and point cloud conversion.")

    def on_fill_curvature_clicked(self):
        if self.ply_file:
            subprocess.run(["python", "fill_curvature.py", "--file", self.ply_file])
        else:
            print("Please import an image and wait for depth estimation and point cloud conversion.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())