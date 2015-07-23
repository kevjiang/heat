#!/usr/bin/python
import numpy as np
from math import ceil, floor, sqrt
import random
import matplotlib.pyplot as plt
import sys
from scipy.cluster.vq import kmeans2, ClusterError
from points import generate_random_points, generate_clustered_points, dataset_generator
from timeit import default_timer
import datetime

NUM_POINTS = 4000
NUM_CLUSTERS = 32
K_CONST = int(ceil(sqrt(NUM_POINTS/2)))
# K_CONST = 5
NUM_ITER = 100
num_iter_counter = 0
MINIT = 'points'

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
    X = generate_clustered_points(0, 420, 0, 280, NUM_POINTS, 30, 40, NUM_CLUSTERS)
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
    heatmap_data = []
    for i, (x, y) in enumerate(centroid):
        weight = 0
        cluster_points = []

        for j, (a, b) in enumerate(points):
            if label[j] == i:
                cluster_points.append((a,b))
                weight += 1

        heatmap_data.append({'value': weight, 'radius': farthest_distance(cluster_points, (x, y)), 'x': x, 'y': y})

    return heatmap_data

def get_heatmap_data():
    #initialize points
    points = init_board(NUM_POINTS)

    print "scipy kmeans2 implementation"

    #try kmeans2 until there are no cluster empty warnings
    centroid, label = None, None
    num_tries = 0
    while centroid is None or label is None:
        num_tries = num_tries + 1
        try:
            centroid, label = kmeans2(points, float(K_CONST), iter=NUM_ITER, minit='random', missing='raise')
        except ClusterError:
            pass

    print "Centroid: " + str(centroid)
    print "Label: " + str(label)
    print "Total # Tries: " + str(num_tries)
    print "K: " + str(K_CONST)

    # plot_points2(points, centroid, label)
    return heatmap_data_converter(points, centroid, label)

def farthest_distance(list_of_points, anchor):
    max_distance = 0
    a = np.array(anchor)

    for (x, y) in list_of_points:
        b = np.array((x, y))
        this_dis = np.linalg.norm(a-b)
        max_distance = max(max_distance, this_dis)

    return int(max_distance)


def average_distance(list_of_points, anchor):
    if len(list_of_points) == 0:
        print '0 array in average_distance function'
        return 0

    total_distance = 0
    a = np.array(anchor)

    for (x, y) in list_of_points:
        b = np.array((x, y))
        total_distance += np.linalg.norm(a-b)

    avg_distance = total_distance/len(list_of_points)
    return int(avg_distance)

def find_distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))


def sigma(points, label, anchor, cluster_num):
    list_of_points = []

    for i, (x, y) in enumerate(points):
        if label[i] == cluster_num:
            list_of_points.append(np.array((x, y)))

    return average_distance(list_of_points, anchor)

# calculates the Davies-Bouldin Index for a clustering
# the Davies-Bouldin Index evaluates intra-cluster similarity and inter-cluster differences
def db_index(points, centroid, label):
    num_clusters = len(centroid)
    running_sum = 0

    for i, ci in enumerate(centroid):
        max_val = 0
        sig_i = sigma(points, label, ci, i)

        for j, cj in enumerate(centroid):
            if (j == i):
                continue
            sig_j = sigma(points, label, cj, j)
            d = find_distance(ci, cj)
            kicker = (sig_i + sig_j) / d
            max_val = max(kicker, max_val)

        running_sum += max_val

    return running_sum / num_clusters

if __name__ == '__main__':

    if len(sys.argv) > 2:
        if sys.argv[1] == '-one': #random generation of a single creative point cluster...the standard way to run
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
                    centroid, label = kmeans2(points, K_CONST, iter=NUM_ITER, minit='points', missing='raise')
                except ClusterError:
                    pass

            print "Centroid: " + str(centroid)
            print "Label: " + str(label)
            print "Total # Tries: " + str(num_tries)
            print "K: " + str(K_CONST)

            plot_points2(points, centroid, label)
            print heatmap_data_converter(points, centroid, label)
            print "Davies-Bouldin Index: " + str(db_index(points, centroid, label) + " (lower values are better)")

            # print centers, clusters, num_iter_counter
            # plot_points(centers, clusters)

        elif (sys.argv[1] == '-sim') and (len(sys.argv) == 4):

            read_file_name = str(sys.argv[2])
            write_file_name = str(sys.argv[3])

            read_file = open(read_file_name, 'r')
            write_file = open(write_file_name, 'w+')

            num_lines_read_file = sum(1 for line in read_file)
            read_file.seek(0)

            tic = default_timer()
            total_num_tries = 0
            total_num_clusters = 0
            total_num_points = 0
            db_running_sum = 0

            for idx, line in enumerate(read_file):
                line_points_data = np.array(eval(line))
                if (len(line_points_data) < 2):  # don't run kmeans if no points in data
                    print 'kmeans skipped because not enough data in this creative'
                    continue

                num_data_points = len(line_points_data)
                k_to_use = int(ceil(sqrt(num_data_points/2)))

                centroid, label = None, None
                num_tries = 0
                while centroid is None or label is None:
                    num_tries = num_tries + 1
                    try:
                        centroid, label = kmeans2(line_points_data, k_to_use, iter=NUM_ITER, minit='points', missing='raise') # minit could be 'random' or 'point'...point gets rid of magnitude of 2 error
                    except ClusterError:
                        pass

                write_file.write('--------------Creative ' + str(idx) + '--------------\n')
                write_file.write("Centroid: " + str(centroid) + '\n')
                write_file.write("Label: " + str(label) + '\n')

                total_num_tries += num_tries
                total_num_clusters += k_to_use
                total_num_points += num_data_points
                db_running_sum += db_index(line_points_data, centroid, label)

                print ('At creative ' + str(idx) + '/' + str(num_lines_read_file))
                print ('Elapsed Time: ' + str(default_timer() - tic))
                print ('Num Data Points: ' + str(num_data_points))
                print ('Num_tries: ' + str(num_tries))

            read_file.close()
            write_file.close()

            toc = default_timer()

            stats_file = open('stats_file.txt', 'a')
            stats_file.write('--------------' + str(datetime.datetime.utcnow()) + '---------------\n')
            stats_file.write('Testing statistics for <' + read_file_name + '>\n')
            stats_file.write('Results file <' + write_file_name + '>\n')
            stats_file.write('Total number of original points: ' + str(total_num_points) + '\n')
            stats_file.write('Total number of original creatives: ' + str(idx) + '\n')
            stats_file.write('Total Elapsed Time: ' + str(toc-tic) + ' seconds \n')
            stats_file.write('Total number of kmeans intiailzation tries: ' + str(total_num_tries) + '\n')
            stats_file.write('Total number of clusters generated: ' + str(total_num_clusters) + '\n')
            stats_file.write('Maximum number of iterations: ' + str(NUM_ITER) + '\n')
            stats_file.write('minit: ' + str(MINIT) + '\n')
            stats_file.write('Average Davies-Bouldin Index: ' + str(db_running_sum/num_lines_read_file) + ' (lower is better)\n')
            stats_file.write('\n')
            stats_file.close()

        elif sys.argv[1] == '-dump' and (len(sys.argv) == 4):  # file, then number of points to be generated
            dataset_generator(str(sys.argv[2]), int(sys.argv[3]))
    else:
        print '-one: standard single cluster generator.  Will plot points.'
        print '-sim <data_file_name> <write_file_name>: simulates reading data from <data_file_name>, dumping clusters into <write_file_name>, and appending statistics to stats_file.txt'
        print '-dump <data_file_name> <num_points>: generates num_points data and writes into data_file_name'
