import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from geopy.distance import geodesic

def two_opt(route, distance_matrix):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue
                new_route = route[:]
                new_route[i:j] = route[j - 1:i - 1:-1]
                if sum(distance_matrix[new_route[k], new_route[k + 1]] for k in range(len(new_route) - 1)) < \
                   sum(distance_matrix[best[k], best[k + 1]] for k in range(len(best) - 1)):
                    best = new_route
                    improved = True
        route = best
    return best

def ant_colony_optimization(pheromones, visibility, num_ants, num_iterations, decay, alpha, beta, distance_matrix):
    num_points = distance_matrix.shape[0]
    best_route = None
    best_distance = float('inf')
    for iteration in range(num_iterations):
        all_routes = []
        all_distances = []
        for _ in range(num_ants):
            route = [0]
            current_point = 0
            for _ in range(1, num_points):
                pheromone = np.power(pheromones[current_point], alpha)
                visib = np.power(visibility[current_point], beta)
                prob = pheromone * visib
                prob[route] = 0
                prob /= prob.sum()
                next_point = np.random.choice(range(num_points), p=prob)
                route.append(next_point)
                current_point = next_point
            route.append(0)
            route = two_opt(route, distance_matrix)
            all_routes.append(route)
            route_distance = sum(distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
            all_distances.append(route_distance)
            if route_distance < best_distance:
                best_route = route
                best_distance = route_distance
        pheromones *= (1 - decay)
        for route, distance in zip(all_routes, all_distances):
            for i in range(len(route) - 1):
                pheromones[route[i], route[i + 1]] += 1 / distance
    return best_route, best_distance

def optimize_delivery_route(dataset_path, num_ants=5, num_iterations=20, alpha=1, beta=2, decay=0.1):
    dataset = pd.read_csv(dataset_path)
    full_distance_matrix = squareform(pdist(dataset[['lat', 'lng']].values, metric=lambda u, v: geodesic(u, v).kilometers))
    depot_distances = np.array([geodesic((row['lat'], row['lng']), (row['depot_lat'], row['depot_lng'])).kilometers for _, row in dataset.iterrows()])
    full_distance_matrix = np.vstack([depot_distances, full_distance_matrix])
    depot_distances_with_depot = np.insert(depot_distances, 0, 0)
    full_distance_matrix = np.column_stack([depot_distances_with_depot, full_distance_matrix])
    pheromones = np.ones((len(full_distance_matrix), len(full_distance_matrix))) * 0.1
    visibility = 1 / (full_distance_matrix + 1e-10)
    best_route, best_distance = ant_colony_optimization(pheromones, visibility, num_ants, num_iterations, decay, alpha, beta, full_distance_matrix)
    best_route_adjusted = [r - 1 for r in best_route[1:-1]]
    dataset['dlvr_seq_num'] = 0
    for idx, route_idx in enumerate(best_route_adjusted):
        dataset.loc[route_idx, 'dlvr_seq_num'] = idx + 1
    output_path = dataset_path.replace('.csv', '_updated.csv')
    output_path = dataset_path.replace('input', 'output')
    dataset.to_csv(output_path, index=False)
    return output_path, best_distance



summary_table = pd.DataFrame(columns=['Dataset', 'Best Route Distance'])
dataset_paths = ['input_datasets\part_a\part_a_input_dataset_1.csv', 'input_datasets\part_a\part_a_input_dataset_2.csv', 
                 'input_datasets\part_a\part_a_input_dataset_3.csv', 'input_datasets\part_a\part_a_input_dataset_4.csv', 
                 'input_datasets\part_a\part_a_input_dataset_5.csv']




for dataset_path in dataset_paths:
    output_path, best_distance = optimize_delivery_route(dataset_path)
    summary_table = summary_table._append({
        'Dataset': dataset_path,
        'Best Route Distance': f"{best_distance:.2f} kms"
    }, ignore_index=True)


summary_table.to_csv('output_datasets\part_a\part_a_best_routes_distance_travelled.csv', index=False)



