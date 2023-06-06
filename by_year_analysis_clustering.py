from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import networkx as nx
import os
import json
from community import community_louvain
import threading
import pandas as pd

max_threads = 16

# Create a semaphore with the maximum number of threads allowed
thread_limiter = threading.Semaphore(max_threads)


def generate_network_by_clustering(
    df, year, features, similarity_threshold, gephi_path, metrics_path
):
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    # Compute the cosine similarity matrix
    similarity_matrix = cosine_similarity(scaled_data)

    # Set the similarity threshold for merging nodes
    merge_threshold = 0.97

    # Use Agglomerative Clustering to identify nodes to be merged
    clustering = AgglomerativeClustering(
        n_clusters=None,
        affinity="precomputed",
        linkage="average",
        distance_threshold=1 - merge_threshold,
    )
    clustering.fit(1 - similarity_matrix)

    # Get the cluster labels (nodes to be merged will have the same label)
    labels = clustering.labels_

    # Merge nodes in the same cluster and compute average features
    merged_features = []
    for label in np.unique(labels):
        if label != -1:
            # Get indices of nodes in this cluster
            indices = np.where(labels == label)[0]

            # Merge nodes by averaging their features
            numeric_features = df.iloc[indices].select_dtypes(
                include=[np.number]
            )
            merged_node = numeric_features.mean().to_dict()
            merged_features.append(merged_node)

    # Convert merged features to DataFrame
    merged_df = pd.DataFrame(merged_features)

    # Calculate similarity for merged nodes
    merged_scaled_data = scaler.transform(merged_df[features])
    merged_similarity_matrix = cosine_similarity(merged_scaled_data)

    # Initialize an empty graph
    song_network = nx.Graph()

    # Add nodes to the graph
    song_network.add_nodes_from(merged_df.index)

    # Add edges between similar songs
    for i in range(len(merged_similarity_matrix)):
        for j in range(i + 1, len(merged_similarity_matrix)):
            if merged_similarity_matrix[i][j] >= similarity_threshold:
                song_network.add_edge(
                    i, j, weight=merged_similarity_matrix[i][j]
                )

    # Add attributes to nodes
    for i, row in merged_df.iterrows():
        song_network.nodes[i]["popularity"] = row["popularity"]
        for feature in features:
            song_network.nodes[i][feature] = row[feature]

    isolated_nodes = [n for n, d in song_network.degree() if d == 0]

    # Cluster nodes using the Louvain method
    partition = community_louvain.best_partition(song_network)

    # Add the community attribute to the nodes
    for i, row in merged_df.iterrows():
        song_network.nodes[i]["community"] = partition[i]

    # Save the network to GEXF file
    if not os.path.exists(gephi_path):
        os.makedirs(gephi_path)

    gexf_file_path = os.path.join(
        gephi_path, f"song_network_clustering_{year}.gexf"
    )
    nx.write_gexf(song_network, gexf_file_path)

    # Calculate network metrics
    metrics = {
        "year": int(year),
        "path": gexf_file_path,
        "number_of_removed_isolated_nodes": len(isolated_nodes),
        "similarity_threshold": similarity_threshold,
        "num_nodes": int(song_network.number_of_nodes()),
        "num_edges": int(song_network.number_of_edges()),
        "avg_clustering_coefficient": nx.average_clustering(song_network),
        "is_connected": nx.is_connected(song_network),
        "number_of_connected_components": int(
            nx.number_connected_components(song_network)
        ),
    }

    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)

    json_file_path = os.path.join(
        metrics_path, f"metrics_clustering_{year}.json"
    )

    with open(json_file_path, "w") as json_file:
        json.dump(metrics, json_file, indent=4)

    return song_network, metrics


def process_year(
    df, year, features, similarity_threshold, gephi_path, metrics_path
):
    gexf_file_path = os.path.join(
        gephi_path, f"song_network_k_nearest_{year}.gexf"
    )

    # Check if the file already exists
    if not os.path.exists(gexf_file_path):
        song_network, metrics = generate_network_by_clustering(
            df, year, features, similarity_threshold, gephi_path, metrics_path
        )
        print(f"Generated K nearest network for {year} with {metrics}.")
    else:
        print(f"File for {year} already exists. Skipping.")


class NetworkGenerationThreadForClustering(threading.Thread):
    def __init__(
        self,
        df,
        year,
        features,
        similarity_threshold,
        gephi_path,
        metrics_path,
    ):
        threading.Thread.__init__(self)
        self.df = df
        self.year = year
        self.features = features
        self.similarity_threshold = similarity_threshold
        self.gephi_path = gephi_path
        self.metrics_path = metrics_path

    def run(self):
        # Acquire the semaphore
        thread_limiter.acquire()

        # Run the process_year function
        process_year(
            self.df,
            self.year,
            self.features,
            self.similarity_threshold,
            self.gephi_path,
            self.metrics_path,
        )

        # Release the semaphore
        thread_limiter.release()
