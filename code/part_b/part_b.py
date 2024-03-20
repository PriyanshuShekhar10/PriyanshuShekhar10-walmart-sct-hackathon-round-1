import pandas as pd
from geopy.distance import geodesic
import numpy as np

dataset_1_path = 'input_datasets\part_b\part_b_input_dataset_1.csv'
dataset_2_path = 'input_datasets\part_b\part_b_input_dataset_2.csv'
dataset_1 = pd.read_csv(dataset_1_path)
dataset_2 = pd.read_csv(dataset_2_path)

def calculate_route(dataset, vehicle_capacity):
    n = len(dataset)
    dataset['dlvr_seq_num'] = np.zeros(n, dtype=int)
    dataset['vehicle_num'] = np.zeros(n, dtype=int)
    vehicle_positions = [tuple(dataset[['depot_lat', 'depot_lng']].iloc[0])] * 2
    visited = np.zeros(n, dtype=bool)
    distances = [0, 0]
    vehicle_loads = [0, 0]
    
    for vehicle_num in range(2):
        while vehicle_loads[vehicle_num] < vehicle_capacity and not all(visited):
            current_position = vehicle_positions[vehicle_num]
            min_dist = None
            next_index = None
            for i in range(n):
                if not visited[i]:
                    order_position = tuple(dataset[['lat', 'lng']].iloc[i])
                    dist = geodesic(current_position, order_position).kilometers
                    if min_dist is None or dist < min_dist:
                        min_dist = dist
                        next_index = i

            if next_index is not None:
                visited[next_index] = True
                vehicle_loads[vehicle_num] += 1
                dataset.at[next_index, 'dlvr_seq_num'] = vehicle_loads[vehicle_num]
                dataset.at[next_index, 'vehicle_num'] = vehicle_num + 1
                vehicle_positions[vehicle_num] = tuple(dataset[['lat', 'lng']].iloc[next_index])
                distances[vehicle_num] += min_dist

            if vehicle_loads[vehicle_num] == vehicle_capacity or all(visited):
                distances[vehicle_num] += geodesic(vehicle_positions[vehicle_num], tuple(dataset[['depot_lat', 'depot_lng']].iloc[0])).kilometers

    return dataset, distances

vehicle_capacity_1 = 20
vehicle_capacity_2 = 30
optimized_dataset_1, distances_1 = calculate_route(dataset_1.copy(), vehicle_capacity_1)
optimized_dataset_2, distances_2 = calculate_route(dataset_2.copy(), vehicle_capacity_2)

optimized_dataset_1.to_csv('output_datasets\part_b\part_b_output_dataset_1.csv', index=False)
optimized_dataset_2.to_csv('output_datasets\part_b\part_b_output_dataset_2.csv', index=False)

total_distances = pd.DataFrame({
    'dataset': ['1', '1', '2', '2'],
    'vehicle_num': [1, 2, 1, 2],
    'distance_travelled': distances_1 + distances_2
})
total_distances.to_csv('output_datasets\part_b\part_b_routes_distance_travelled.csv', index=False)
