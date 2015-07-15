import numpy as np
from random import randint
import matplotlib.pyplot as plt

def generate_random_points(minX, maxX, minY, maxY, num_points):
    data = []
    for _ in range(num_points):
        data.append([randint(minX, maxX), randint(minY, maxY)])
    return data

def generate_clustered_points(minX, maxX, minY, maxY, num_points, cluster_width, cluster_height, num_clusters):
    data = []
    for _ in range(num_clusters):
        x_floor = randint(minX, maxX)
        y_floor = randint(minY, maxY)
        x_ceil = x_floor + cluster_width
        y_ceil = y_floor + cluster_height
        cluster = generate_random_points(x_floor, x_ceil, y_floor, y_ceil, num_points/(num_clusters*2))
        data += cluster

    #for testing only
    center_xs, center_ys = zip(*data)
    plt.subplot(211)
    plt.title('Clusters Generated')
    plt.scatter(center_xs, center_ys)
    # plt.show()

    data = data + generate_random_points(minX, maxX, minY, maxY, num_points/2)
    return np.array(data)
