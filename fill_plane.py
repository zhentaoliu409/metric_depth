#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Liu Zhentao
# @Time    : 2025/1/10 13:23
# Author  : Liu Zhentao
# File    : fill_plane.py
# Licensed under the MIT License
import argparse
import copy
import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import KDTree


if __name__ == '__main__':
    # adjust view to XOY plane and show points cloud
    def draw_geometries(geometries, window_name):
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name)
        for geometry in geometries:
            vis.add_geometry(geometry)

        # Set view to XOY plane
        ctr = vis.get_view_control()
        all_points = np.vstack([np.asarray(pcd.points) for pcd in geometries])
        centroid = np.mean(all_points, axis=0)
        ctr.set_front([0, 0, -1])
        ctr.set_lookat(centroid)
        ctr.set_up([0, -1, 0])
        ctr.set_zoom(0.8)

        vis.run()
        vis.destroy_window()
        return 0

    def smooth_point_cloud(pcd, voxel_size=0.01, nb_neighbors=20, std_ratio=10.0):
        # Voxel downsampling
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        # Estimate normals
        downsampled_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Remove statistical outliers
        cl, ind = downsampled_pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

        # Select inliers
        inlier_cloud = downsampled_pcd.select_by_index(ind)

        return inlier_cloud

    def densiy_and_distance(pcd):
        points = np.asarray(pcd.points)
        # Construct a KDTree
        kdtree = KDTree(points)
        # Query the nearest neighbors
        distances, _ = kdtree.query(points, k=2)
        # Calculate the mean distance
        mean_distance = np.mean(distances[:, 1])

        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        volume = np.prod(max_bound - min_bound)

        # Calculate the number of points per unit volume
        num_points = len(points)
        density = num_points / volume
        print(f"Number of points: {num_points}, Density: {density}, mean distance: {mean_distance}")
        return mean_distance, density

    # define a function to fit plane by Least Squares Method
    def generate_plane(point_cloud):

        # Extract points from the point cloud
        points = np.asarray(point_cloud.points)

        # Compute the centroid of the points
        centroid = np.mean(points, axis=0)

        # Compute the covariance matrix
        cov_matrix = np.cov(points - centroid, rowvar=False)

        # Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # The normal vector is the eigenvector corresponding to the smallest eigenvalue
        normal_vector = eigenvectors[:, 0]

        # Compute the plane offset (D) using the centroid
        D = -normal_vector.dot(centroid)

        # Compute the minimum bounding box of the point cloud
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)

        density = 100000
        # Generate random points within the bounding box
        random_points = np.random.rand(density, 3) * (max_bound - min_bound) + min_bound
        # Project the random points onto the plane
        distances = (random_points @ normal_vector + D) / np.linalg.norm(normal_vector)
        plane_points = random_points - np.outer(distances, normal_vector)

        # Filter the plane points to be within the bounding box of the original point cloud
        within_bounds = np.all((plane_points >= min_bound) & (plane_points <= max_bound), axis=1)
        plane_points = plane_points[within_bounds]

        plane_pcd = o3d.geometry.PointCloud()
        plane_pcd.points = o3d.utility.Vector3dVector(plane_points)

        # Apply Moving Least Squares (MLS) smoothing
        plane_pcd = plane_pcd.voxel_down_sample(voxel_size=0.01)
        plane_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        plane_equation = f"{normal_vector[0]}x + {normal_vector[1]}y + {normal_vector[2]}z + {D} = 0"
        # print(f"Plane equation: {plane_equation}")

        # Set the color of the plane to gray
        plane_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        return plane_pcd, normal_vector

    # The function is coloured and divided into inner and outer points based on the distance between point clouds
    def colorize_distance_and_segment(image_rotate, normal_vector, plane, threshold, mode=0):
        # Extract the points from the rotated point cloud
        points = np.asarray(image_rotate.points)

        # Compute the plane offset (D)
        D = -normal_vector.dot(np.squeeze(np.asarray(plane.points)[0]))

        # Calculate the distances from the points to the plane

        distances = ((points @ normal_vector + D) / np.linalg.norm(normal_vector))
        colors = np.zeros((points.shape[0], 3))
        # Normalize the distances to the range 0-255
        distances_normalized = 255 * (np.abs(distances) - np.abs(distances).min()) / (
                    np.abs(distances).max() - np.abs(distances).min())


        threshold = threshold * 255

        # Split the points into inliers and outliers
        if mode == 1:

            outliers_indices = np.where(distances > 0)[0]
            inliers_indices = np.where(distances <= 0)[0]

        elif mode == 2:

            outliers_indices = np.where((distances > 0) & (distances_normalized > threshold))[0]
            inliers_indices = np.where((distances >= 0) | (distances_normalized < threshold))[0]
        else:

            outliers_indices = np.where(distances_normalized > threshold)[0]
            inliers_indices = np.where(distances_normalized <= threshold)[0]

        # Create inliers and outliers point clouds
        inliers = image_rotate.select_by_index(inliers_indices)
        outliers = image_rotate.select_by_index(outliers_indices)
        # Set the colors based on the normalized distances
        colors[:, 0] = distances_normalized  # Set the red channel based on the distances
        colors[:, 1] = distances_normalized  # Set the green channel based on the distances
        colors[:, 2] = distances_normalized  # Set the blue channel based on the distances
        # Apply the colors to the point cloud
        rotate_clone = copy.deepcopy(image_rotate)
        rotate_clone.colors = o3d.utility.Vector3dVector(colors / 255.0)

        return inliers, outliers, rotate_clone

    def iterative_plane_fitting(point_cloud, max_iterations=100, initial_threshold=0.1, threshold_adjustment=0.01):

        best_threshold = initial_threshold
        min_change_ratio = float('inf')
        origin_cloud = copy.deepcopy(point_cloud)
        previous_inliers_count = 0
        stable_iterations = 1

        for threshold in np.arange(0.1, 0.90, 0.01):
            plane, normal_vector = generate_plane(point_cloud)
            inliers, outliers, _ = colorize_distance_and_segment(origin_cloud, normal_vector, plane, threshold)
            current_inliers_count = len(inliers.points)
            if previous_inliers_count == 0:
                change_ratio = 1
            else:
                change_ratio = abs(current_inliers_count - previous_inliers_count) / current_inliers_count

            if change_ratio < min_change_ratio:
                min_change_ratio = change_ratio
                best_threshold = threshold
                stable_iterations = 1
            else:
                stable_iterations += 1

            previous_inliers_count = current_inliers_count
            print(
                f"Best threshold: {best_threshold}, Threshold: {threshold}, Inliers count: {current_inliers_count}, Change ratio: {change_ratio}")

            if change_ratio < 0.01 and change_ratio != 0:
                best_threshold = best_threshold - threshold_adjustment
                break
            if stable_iterations >= 8:
                break

        optimal_threshold = best_threshold
        # Perform preliminary search for the optimal threshold
        print(f"Optimal threshold determined: {optimal_threshold}")

        point_cloud = origin_cloud
        previous_outliers_count = 0

        for i in range(max_iterations):
            plane, normal_vector = generate_plane(point_cloud)
            inliers, outliers, colorize = colorize_distance_and_segment(origin_cloud, normal_vector, plane,
                                                                        optimal_threshold)
            current_outliers_count = len(outliers.points)
            if previous_outliers_count == 0:
                change_ratio = 1
            else:
                change_ratio = abs(current_outliers_count - previous_outliers_count) / current_outliers_count

            if change_ratio < 1e-05 and change_ratio != 0:
                break
            point_cloud = inliers
            previous_outliers_count = current_outliers_count

        plane, normal_vector = generate_plane(point_cloud)
        inliers, outliers, colorize = colorize_distance_and_segment(origin_cloud, normal_vector, plane,
                                                                    optimal_threshold, mode=2)
        draw_geometries([origin_cloud, plane], window_name=f" Point Cloud")
        draw_geometries([colorize], window_name=f"colorize")
        draw_geometries([outliers], window_name=f"outliers")
        return inliers, outliers, plane, normal_vector

    # This function fills the points according to the distance of the point cloud from its upper boundary (the fitted road surface))
    def filled_pothole(pcd, normal_vector, plane_point):
        global mean_distance

        # Extract points and colors from the point cloud
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        # Compute the plane offset (D)
        D = -normal_vector.dot(np.squeeze(np.asarray(plane_point.points)[0]))

        # Calculate the distances from the points to the plane
        distances = (points @ normal_vector + D) / np.linalg.norm(normal_vector)

        # Determine the maximum distance (depth)
        depth = distances.max()

        # Generate filled points based on the distances
        filled_points = []
        for point, distance in zip(points, distances):
            num_fill_points = int(distance / mean_distance)  # Adjust the step size as needed
            for i in range(num_fill_points):
                filled_points.append(point - i * mean_distance * normal_vector)

        # Convert the filled points to a point cloud
        filled_points = np.array(filled_points)
        filled_pcd = o3d.geometry.PointCloud()
        filled_pcd.points = o3d.utility.Vector3dVector(filled_points)

        # Calculate the average color of the inliers
        average_color = np.mean(colors, axis=0)
        filled_pcd.paint_uniform_color(average_color)
        return filled_pcd, depth

    # This function uses slicing to calculate the volume of the filled point cloud.
    def compute_volume_slicing(pcd, slice_thickness=0.01):
        points = np.asarray(pcd.points)
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        volume = 0.0

        y_slices = np.arange(min_bound[1], max_bound[1], slice_thickness)
        for y in y_slices:
            slice_points = points[(points[:, 1] >= y) & (points[:, 1] < y + slice_thickness)]
            if len(slice_points) >= 3:
                hull = ConvexHull(slice_points[:, [0, 2]])
                area = hull.volume
                volume += area * slice_thickness

        return volume

    parser = argparse.ArgumentParser(description="Process point cloud data.")
    parser.add_argument("--file", type=str, required=True, help="Path to the point cloud file")
    args = parser.parse_args()

    # Load Point Cloud
    point_cloud = o3d.io.read_point_cloud(args.file)
    draw_geometries([point_cloud], window_name="Oriented Bounding Box")

    point_rotate = smooth_point_cloud(point_cloud)
    draw_geometries([point_rotate], window_name="Smoothed Point Cloud")

    # Calculate the minimum distance between points
    mean_distance, density = densiy_and_distance(point_rotate)

    # Delineation of inner and outer points
    inliers, outliers, plane, normal_vector = iterative_plane_fitting(point_rotate)

    # fill the potholes
    pcd_fill, depth = filled_pothole(outliers, normal_vector, plane)
    draw_geometries([pcd_fill], window_name="Filled block")
    draw_geometries([pcd_fill, point_rotate], window_name="Filled Pothole")

    # Output depth and volume
    print(f"Estimated depth: {depth} m")
    volume1 = compute_volume_slicing(pcd_fill)
    print(f"Estimated volume (slicing): {volume1} m³")

