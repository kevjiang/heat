#!/usr/bin/python
import numpy as np
from random import randint
import matplotlib.pyplot as plt
import sys

def generate_random_points(minX, maxX, minY, maxY, num_points):
    data = []
    for _ in range(num_points):
        data.append([float(randint(minX, maxX)), float(randint(minY, maxY))])
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

#generates a random amount of data for num_creatives with total_points data
def dataset_generator(write_file_name, total_points):
    points_count = 0
    creatives = []

    #loop until points_count < total_points
    #within loop, call either generate_clustered_points or generate_random_points with random parameters
    #append data to creatives list

    while points_count < total_points:
        num_points = randint(250000, 250001)
        temp = generate_random_points(0, 420, 0, 280, num_points)
        creatives.append(temp)
        points_count += num_points

    clicks_data_file = open(write_file_name, 'w+')

    for data in creatives:
        clicks_data_file.write(str(data) + '\n')

    clicks_data_file.close()
    #afterwards, write data in creatives, line by line, into clicks_data.txt

if __name__ == '__main__':
    dataset_generator('db_xl.txt', 2000000)
