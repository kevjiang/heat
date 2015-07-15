#!/usr/bin/python
import numpy as np
from math import ceil, floor, sqrt
import random
import matplotlib.pyplot as plt
import sys
from scipy.cluster.vq import kmeans2, ClusterError
from points import generate_random_points, generate_clustered_points

NUM_POINTS = 4000
NUM_CLUSTERS = 32
K_CONST = int(ceil(sqrt(NUM_POINTS/2)))
# K_CONST = 5
NUM_ITER = 1000
num_iter_counter = 0

def cluster_points(X, mu):
    clusters = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]]))
                        for i in enumerate(mu)], key=lambda t: t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters


def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis=0))
    return newmu


def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))


def find_centers(X, K):
    global num_iter_counter
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not(has_converged(mu, oldmu)) and num_iter_counter < NUM_ITER:
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
        num_iter_counter = num_iter_counter + 1
    return(mu, clusters)


def init_board(N):
    # X = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(N)])
    X = generate_clustered_points(0, 420, 0, 240, NUM_POINTS, 30, 40, NUM_CLUSTERS)
    return X


def plot_points(centers, clusters):
    #generate random color sequence for clusters'
    # colors = np.random.rand(K_CONST)
    colors = ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(K_CONST)]

    #process points for plotting, add points to scatter, and show plot
    center_xs, center_ys = zip(*centers)
    for key, value in clusters.iteritems():
        color = colors[key]
        xs_to_plot = []
        ys_to_plot = []
        for (x, y) in value:
            xs_to_plot.append(x)
            ys_to_plot.append(y)
        plt.scatter(xs_to_plot, ys_to_plot, c=[color]*len(xs_to_plot), s=20)

    plt.scatter(center_xs, center_ys, c=colors, s=100)
    plt.show()


def plot_points2(points, centroid, label):
    colors = ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(K_CONST)]
    center_xs, center_ys = zip(*centroid)

    plt.title('Centroid/Cluster Overlay')
    plt.scatter(center_xs, center_ys, c=colors, s=100)

    plt.subplot(212)
    plt.title('Points')

    for counter, (x, y) in enumerate(points):
        plt.scatter(x, y, c=colors[label[counter]], s=20)

    plt.show()


def heatmap_data_converter(points, centroid, label):
    #should be of form (x,y), weight, radius
    heatmap_data = {}
    for i, (x, y) in enumerate(centroid):
        weight = 0
        cluster_points = []

        for j, (a, b) in enumerate(points):
            if label[j] == i:
                cluster_points.append((a,b))
                weight += 1

        heatmap_data[(x, y)] = {'weight': weight, 'radius': farthest_distance(cluster_points, (x, y))}

    return heatmap_data


def farthest_distance(list_of_points, anchor):
    max_distance = 0
    a = np.array(anchor)

    for (x, y) in list_of_points:
        b = np.array((x, y))
        this_dis = np.linalg.norm(a-b)
        max_distance = max(max_distance, this_dis)

    return int(max_distance)


def average_distance(list_of_points, anchor):
    total_distance = 0
    a = np.array(anchor)

    for (x, y) in list_of_points:
        b = np.array((x, y))
        total_distance += np.linalg.norm(a-b)

    avg_distance = total_distance/len(list_of_points)
    return int(avg_distance)

if __name__ == '__main__':
    #initialize points
    points = init_board(NUM_POINTS)
    print points

    #run k-means and find list of centers and corresponding clusters
    # print "Old kmeans implementation"
    # centers, clusters = find_centers(points, K_CONST)
    # print "Number of Iterations: " + num_iter_counter

    print "scipy kmeans2 implementation"

    #try kmeans2 until there are no cluster empty warnings
    centroid, label = None, None
    num_tries = 0
    while centroid is None or label is None:
        num_tries = num_tries + 1
        try:
            centroid, label = kmeans2(points, K_CONST, iter=NUM_ITER, minit='random', missing='raise')
        except ClusterError:
            pass

    print "Centroid: " + str(centroid)
    print "Label: " + str(label)
    print "Total # Tries: " + str(num_tries)
    print "K: " + str(K_CONST)

    plot_points2(points, centroid, label)
    print heatmap_data_converter(points, centroid, label)

    # print centers, clusters, num_iter_counter
    # plot_points(centers, clusters)
