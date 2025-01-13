#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Liu Zhentao
# @Time    : 2025/1/10 13:23
# Author  : Liu Zhentao
# File    : 3dpoints.py
# Licensed under the MIT License
import open3d as o3d
import numpy as np

def rotate(pcd):
    # 创建旋转矩阵（沿 x 轴旋转 π弧度即180°）
    R = pcd.get_rotation_matrix_from_axis_angle(np.array([np.pi, 0, 0]))
    # 应用旋转
    pcd.rotate(R, center=(0, 0, 0))

    # 使用 RANSAC 检测平面
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    [a, b, c, d] = plane_model

    # 计算旋转矩阵，使得平面法向量与 Z 轴对齐
    normal_vector = np.array([a, b, c])
    target_axis = np.array([0, 1, 0])
    # 计算旋转轴和旋转角度
    v = np.cross(normal_vector, target_axis)  # 旋转轴
    s = np.linalg.norm(v)                # 旋转轴的模
    c = np.dot(normal_vector, target_axis)    # 旋转角的余弦值

    # 构建旋转矩阵（使用罗德里格斯公式）
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

    # 应用旋转矩阵
    pcd.rotate(R, center=(0, 0, 0))
    return pcd

def align_wall_to_xz(points):
    # 计算点云的中心
    centroid = np.mean(points, axis=0)

    # 将点云移到原点
    centered_points = points - centroid

    # 计算点云的协方差矩阵
    cov_matrix = np.cov(centered_points.T)

    # 计算特征值和特征向量
    eig_values, eig_vectors = np.linalg.eig(cov_matrix)

    # 找到最大的特征向量（主方向）
    main_direction = eig_vectors[:, np.argmax(eig_values)]

    # 计算旋转角度，使主方向与 Z 轴对齐
    target_direction = np.array([1, 0, 1])
    rotation_axis = np.cross(main_direction, target_direction)
    angle = np.arccos(np.clip(np.dot(main_direction, target_direction), -1.0, 1.0))

    # 如果旋转轴不为零，进行旋转
    if np.linalg.norm(rotation_axis) > 0.001:
        rotation_axis /= np.linalg.norm(rotation_axis)  # 单位化旋转轴
        rotation_vector = rotation_axis * angle
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector)

        # 旋转点云
        rotated_points = centered_points @ rotation_matrix.T

        return rotated_points + centroid  # 返回到原始位置

pcd = o3d.io.read_point_cloud("outputs/image.ply")
aligned_points = align_wall_to_xz(np.asarray(pcd.points))
pcd.points = o3d.utility.Vector3dVector(aligned_points)
# 显示点云
o3d.visualization.draw_geometries([pcd])
