#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Liu Zhentao
# @Time    : 2025/1/10 13:23
# Author  : Liu Zhentao
# File    : fill_curvature.py
# Licensed under the MIT License
import argparse
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
        return mean_distance,density

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
        #print(f"Plane equation: {plane_equation}")

        # Set the color of the plane to gray
        plane_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        return plane_pcd, normal_vector

    # This function precomputes the neighbors of each point in the point cloud
    def precompute_neighbors(point_cloud, search_radius):
        kdtree = o3d.geometry.KDTreeFlann(point_cloud)
        neighbors = []
        for i in range(len(point_cloud.points)):
            [_, idx, _] = kdtree.search_radius_vector_3d(point_cloud.points[i], search_radius)
            neighbors.append(idx)
        return neighbors

    # This function computes the curvature of a point based on its neighbors
    def compute_curvature(neighbors, points, i):
        idx = neighbors[i]
        if len(idx) < 3:
            return 0
        neighbors_points = points[idx, :]
        covariance_matrix = np.cov(neighbors_points.T)
        eigenvalues, _ = np.linalg.eigh(covariance_matrix)
        curvature = eigenvalues[0] / np.sum(eigenvalues)
        return curvature

    # This function estimates the curvature of the point cloud
    def estimate_curvature(point_cloud):
        global mean_distance
        # Set the search radius for estimating normals( which defines the neighborhood of each point )
        search_radius = 30 * mean_distance
        # Estimate normals
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=30))

        # Precompute neighbors
        neighbors = precompute_neighbors(point_cloud, search_radius)
        points = np.asarray(point_cloud.points)

        # Compute curvature sequentially
        curvatures = [compute_curvature(neighbors, points, i) for i in range(len(points))]

        return np.array(curvatures)

    # This function segments the point cloud based on curvature
    def segment_by_curvature(point_cloud, curvatures, curvature_threshold=0.005):
        inliers_indices = np.where(curvatures <= curvature_threshold)[0]
        outliers_indices = np.where(curvatures > curvature_threshold)[0]
        print(f"Number of inliers: {len(inliers_indices)}, Number of outliers: {len(outliers_indices)}")

        inliers = point_cloud.select_by_index(inliers_indices)
        outliers = point_cloud.select_by_index(outliers_indices)
        return inliers, outliers

    # This function finds the optimal curvature threshold for segmenting the point cloud
    def find_optimal_curvature_threshold(point_cloud, curvatures, initial_threshold=0.001, step=0.0001, max_iterations=100):
        best_threshold = initial_threshold
        min_change_ratio = float('inf')
        previous_outliers_count = 0
        threshold = best_threshold

        stable_iterations = 0
        for i in range(max_iterations):
            inliers, outliers = segment_by_curvature(point_cloud, curvatures, threshold)
            current_outliers_count = len(outliers.points)
            if previous_outliers_count == 0:
                change_ratio = 1
            else:
                change_ratio = abs(current_outliers_count - previous_outliers_count) / current_outliers_count
            print(f"Iteration {i + 1}: Previous outliers: {previous_outliers_count}, Current outliers: {current_outliers_count}, Change ratio: {change_ratio}, Threshold: {threshold}, Best threshold: {best_threshold}")

            if change_ratio < min_change_ratio:
                min_change_ratio = change_ratio
                best_threshold = threshold
                stable_iterations = 0  # Reset stable iterations counter
            else:
                stable_iterations += 1

            if change_ratio < 0.01 and change_ratio != 0:
                break
            if stable_iterations >= 10:
                break

            previous_outliers_count = current_outliers_count
            threshold = threshold + step
        print(f"Optimal threshold determined: {best_threshold}")
        inliers, outliers = segment_by_curvature(point_cloud, curvatures, best_threshold)
        return best_threshold,inliers,outliers

    def extract_hollow_region(outliers, original_pcd):
        # Fit a plane to the outliers (pothole boundary)
        plane_pcd, normal_vector = generate_plane(outliers)

        # Identify points above the fitted plane
        points = np.asarray(original_pcd.points)
        D = -normal_vector.dot(np.squeeze(np.asarray(plane_pcd.points)[0]))
        distances = (points @ normal_vector + D) / np.linalg.norm(normal_vector)
        above_plane_indices = np.where(distances > 0)[0]
        above_plane_points = np.asarray(original_pcd.select_by_index(above_plane_indices).points)
        # Remove points that are part of the pothole boundary
        outlier_points = np.asarray(outliers.points)
        outlier_set = set(map(tuple, outlier_points))

        hollow_region_indices = []
        for i, point in enumerate(above_plane_points):
            if tuple(point) not in outlier_set:
                hollow_region_indices.append(above_plane_indices[i])
        combined_indices = np.union1d(hollow_region_indices, above_plane_indices)

        # Select the hollow region points from the original point cloud
        hollow_region_pcd = original_pcd.select_by_index(combined_indices)
        road_pcd = original_pcd.select_by_index(combined_indices, invert=True)
        road_indices = np.setdiff1d(np.arange(len(points)), combined_indices)
        # Set the color of the hollow region to white，and the color of the road to gray
        color = np.asarray(original_pcd.colors)
        color[combined_indices] = [1, 1, 1]
        color[road_indices] = [0.5, 0.5, 0.5]
        original_pcd.colors = o3d.utility.Vector3dVector(color)

        return hollow_region_pcd, plane_pcd, normal_vector, road_pcd, original_pcd

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

    # Segment the point cloud based on curvature
    curvatures = estimate_curvature(point_rotate)
    optimal_threshold, inliers, outliers = find_optimal_curvature_threshold(point_rotate, curvatures)
    draw_geometries([inliers], window_name="Inliers (Road Surface)")
    draw_geometries([outliers], window_name="Outliers (Potholes or Protrusions)")

    hollow, plane, normal_vector, road, color = extract_hollow_region(outliers, point_rotate)
    draw_geometries([point_rotate, plane], window_name="Point Cloud with Plane")
    draw_geometries([color], window_name="color")
    draw_geometries([hollow], window_name="Hollow Region")

    # fill the potholes
    pcd_fill, depth = filled_pothole(hollow, normal_vector, plane)
    draw_geometries([pcd_fill], window_name="Filled block")
    draw_geometries([pcd_fill, road], window_name="Filled Pothole")

    # Output depth and volume
    print(f"Estimated depth: {depth} m")
    volume1 = compute_volume_slicing(pcd_fill)
    print(f"Estimated volume (slicing): {volume1} m³")
