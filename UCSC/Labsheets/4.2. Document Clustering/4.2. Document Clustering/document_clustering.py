# -*- coding: utf-8 -*-
"""
Created July 2017

@author: aw
"""

from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.cluster import AffinityPropagation
from matplotlib.font_manager import FontProperties
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
from utils import build_feature_matrix
from normalization import normalize_corpus
import pandas as pd
import numpy as np

# Load the movie data from local file
movie_data = pd.read_csv(
    'G:/Github/NLP/UCSC/Labsheets/4.2. Document Clustering/4.2. Document Clustering/movie_data.csv')
# View the beginning to ensure all is ok
print(movie_data.head())

movie_titles = movie_data['Title'].tolist()
movie_synopses = movie_data['Synopsis'].tolist()

# Show the first movie title and synopsis
print('Movie:', movie_titles[0])
print('Movie Synopsis:', movie_synopses[0][:1000])


# Normalize the corpus as usual
norm_movie_synopses = normalize_corpus(movie_synopses,
                                       lemmatize=True,
                                       only_text_chars=True)

# Extract (tf-idf) features using out utils library
vectorizer, feature_matrix = build_feature_matrix(norm_movie_synopses,
                                                  feature_type='tfidf',
                                                  min_df=0.24, max_df=0.85,
                                                  ngram_range=(1, 2))
# How many features do we have?
print(feature_matrix.shape)

# Get the names of the features
feature_names = vectorizer.get_feature_names()

# Show the first twenty features (check more if needed)
print(feature_names[:20])


# Define the k-means clustering function which will call sklearn's KMeans function
# with our parameters

def k_means(feature_matrix, num_clusters=5):
    km = KMeans(n_clusters=num_clusters,
                max_iter=10000)
    km.fit(feature_matrix)
    clusters = km.labels_
    return km, clusters


# We will assume that we want the 100 movies divided into 5 clusters (try other values)
num_clusters = 5
# Call the above function to get clusters and assign the cluster labels to the movies
km_obj, clusters = k_means(feature_matrix=feature_matrix,
                           num_clusters=num_clusters)
movie_data['Cluster'] = clusters


# Get the total number of movies per cluster
c = Counter(clusters)
print(c.items())

# We define a generic function to extract details of the clusters we get (from each of our algorithms)


def get_cluster_data(clustering_obj, movie_data,
                     feature_names, num_clusters,
                     topn_features=10):

    cluster_details = {}
    # Get cluster centroids
    ordered_centroids = clustering_obj.cluster_centers_.argsort()[:, ::-1]
    # Get key features for each cluster
    # Get movies belonging to each cluster
    for cluster_num in range(num_clusters):
        cluster_details[cluster_num] = {}
        cluster_details[cluster_num]['cluster_num'] = cluster_num
        key_features = [feature_names[index]
                        for index
                        in ordered_centroids[cluster_num, :topn_features]]
        cluster_details[cluster_num]['key_features'] = key_features

        movies = movie_data[movie_data['Cluster']
                            == cluster_num]['Title'].values.tolist()
        cluster_details[cluster_num]['movies'] = movies

    return cluster_details


# We also define a simple function to print the data we extracted with the above function
def print_cluster_data(cluster_data):
    # Print cluster details
    for cluster_num, cluster_details in cluster_data.items():
        print('Cluster {} details:'.format(cluster_num))
        print('-'*20)
        print('Key features:', cluster_details['key_features'])
        print('Movies in this cluster:')
        print(', '.join(cluster_details['movies']))
        print('='*40)


# We want to be able to visualize the clusters that we got
# Since these will mostly be for multidimensional data, we need to reduce their dimensionality
# We use sklearn's multidimensional scaling (MDS) algorithm to do this

# This is not a regular function - it has no return value, but only a 'side effect' (plotting)
# This is a generic function to plot the output of any clustering algorithm with dynamic # of clusters

def plot_clusters(num_clusters, feature_matrix,
                  cluster_data, movie_data,
                  plot_size=(16, 8)):
    # Generate random colors for clusters
    def generate_random_color():
        color = '#%06x' % random.randint(0, 0xFFFFFF)
        return color
    # Define different markers for clusters
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']
    # Build the cosine distance matrix
    cosine_distance = 1 - cosine_similarity(feature_matrix)
    # Reduce the dimensionality (in this case to 2) using MDS
    mds = MDS(n_components=2, dissimilarity="precomputed",
              random_state=1)
    # Get the coordinates of clusters in the new low-dimensional (2-dim) space
    plot_positions = mds.fit_transform(cosine_distance)
    x_pos, y_pos = plot_positions[:, 0], plot_positions[:, 1]
    # Set up the cluster plotting data
    cluster_color_map = {}
    cluster_name_map = {}
    for cluster_num, cluster_details in cluster_data.items():
        # Assign cluster features to unique label
        cluster_color_map[cluster_num] = generate_random_color()
        cluster_name_map[cluster_num] = ', '.join(
            cluster_details['key_features'][:5]).strip()
    # Map each unique cluster label with its coordinates and movies
    cluster_plot_frame = pd.DataFrame({'x': x_pos,
                                       'y': y_pos,
                                       'label': movie_data['Cluster'].values.tolist(),
                                       'title': movie_data['Title'].values.tolist()
                                       })
    grouped_plot_frame = cluster_plot_frame.groupby('label')
    # Set the figure size and axes of the plot
    fig, ax = plt.subplots(figsize=plot_size)
    ax.margins(0.05)
    # Plot each cluster with its co-ordinates and the corresponding movie title
    for cluster_num, cluster_frame in grouped_plot_frame:
        marker = markers[cluster_num] if cluster_num < len(markers) \
            else np.random.choice(markers, size=1)[0]
        ax.plot(cluster_frame['x'], cluster_frame['y'],
                marker=marker, linestyle='', ms=12,
                label=cluster_name_map[cluster_num],
                color=cluster_color_map[cluster_num], mec='none')
        ax.set_aspect('auto')
        ax.tick_params(axis='x', which='both', bottom='off', top='off',
                       labelbottom='off')
        ax.tick_params(axis='y', which='both', left='off', top='off',
                       labelleft='off')
    fontP = FontProperties()
    fontP.set_size('small')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01), fancybox=True,
              shadow=True, ncol=5, numpoints=1, prop=fontP)
    # Add labels as necessary
    for index in range(len(cluster_plot_frame)):
        ax.text(cluster_plot_frame.loc[index]['x'],
                cluster_plot_frame.loc[index]['y'],
                cluster_plot_frame.loc[index]['title'], size=8)
    # Finally show the plot (only when this is called will the plot be displayed)
    plt.show()


# Analyse the k-means clustering data
cluster_data = get_cluster_data(clustering_obj=km_obj,
                                movie_data=movie_data,
                                feature_names=feature_names,
                                num_clusters=num_clusters,
                                topn_features=5)

# Print the detailed clustering analysis results
print_cluster_data(cluster_data)

# Now visualize the clusters produced
plot_clusters(num_clusters=num_clusters,
              feature_matrix=feature_matrix,
              cluster_data=cluster_data,
              movie_data=movie_data,
              plot_size=(16, 8))


# We now repeat the steps of performing clustering using a different algorithm: AP


def affinity_propagation(feature_matrix):

    sim = feature_matrix * feature_matrix.T
    sim = sim.todense()
    ap = AffinityPropagation()
    ap.fit(sim)
    clusters = ap.labels_
    return ap, clusters


# Get clusters using affinity propagation (AP)
ap_obj, clusters = affinity_propagation(feature_matrix=feature_matrix)
movie_data['Cluster'] = clusters

# Get the total number of movies per cluster as before
c = Counter(clusters)
print(c.items())

# How many clusters are there in total?
total_clusters = len(c)
print('Total Clusters:', total_clusters)

# Get clustering analysis data using our generic function
cluster_data = get_cluster_data(clustering_obj=ap_obj,
                                movie_data=movie_data,
                                feature_names=feature_names,
                                num_clusters=total_clusters,
                                topn_features=5)
# Print clustering analysis results
print_cluster_data(cluster_data)

# Visualize the clusters as before using our generic function
plot_clusters(num_clusters=num_clusters,
              feature_matrix=feature_matrix,
              cluster_data=cluster_data,
              movie_data=movie_data,
              plot_size=(16, 8))


# We finally repeat the steps of performing clustering again for Hierarchical clustering


def ward_hierarchical_clustering(feature_matrix):

    cosine_distance = 1 - cosine_similarity(feature_matrix)
    linkage_matrix = ward(cosine_distance)
    return linkage_matrix

# Plotting hierarchical plot objects is different to 'partitioned' (hard) clusters
# since they need to be plotted as a dendrogram and not points on a 2-dim graph


def plot_hierarchical_clusters(linkage_matrix, movie_data, figure_size=(8, 12)):
    # Set the size of the plot using pyplot (from matplotlib)
    fig, ax = plt.subplots(figsize=figure_size)
    movie_titles = movie_data['Title'].values.tolist()
    # Plot the dendrogram for the given plot object
    ax = dendrogram(linkage_matrix, orientation="left", labels=movie_titles)
    plt.tick_params(axis='x',
                    which='both',
                    bottom='off',
                    top='off',
                    labelbottom='off')
    plt.tight_layout()
    plt.savefig('ward_hierachical_clusters.png', dpi=200)


# Build ward's linkage matrix by calling the above function
linkage_matrix = ward_hierarchical_clustering(feature_matrix)
# Plot the clustering output as a dendrogram
plot_hierarchical_clusters(linkage_matrix=linkage_matrix,
                           movie_data=movie_data,
                           figure_size=(8, 10))
