import os
import json
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from community import community_louvain

import threading

max_threads = 8

# Create a semaphore with the maximum number of threads allowed
thread_limiter = threading.Semaphore(max_threads)


def generate_network_by_year(
    df, year, features, similarity_threshold, gephi_path, metrics_path
):
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    # Compute the cosine similarity matrix
    similarity_matrix = cosine_similarity(scaled_data)

    # Initialize an empty graph
    song_network = nx.Graph()

    # Add nodes to the graph
    song_network.add_nodes_from(df.index)

    # Add edges between similar songs
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i][j] >= similarity_threshold:
                song_network.add_edge(i, j, weight=similarity_matrix[i][j])

    # Add attributes to nodes
    for i, row in df.iterrows():
        song_network.nodes[i]["popularity"] = row["popularity"]
        for feature in features:
            song_network.nodes[i][feature] = row[feature]

    # Remove isolated nodes
    isolated_nodes = [n for n, d in song_network.degree() if d == 0]
    song_network.remove_nodes_from(isolated_nodes)
    df.drop(isolated_nodes, inplace=True)

    # Cluster nodes using the Louvain method
    partition = community_louvain.best_partition(song_network)

    # Add the community attribute to the nodes
    for i, row in df.iterrows():
        song_network.nodes[i]["community"] = partition[i]

    # Save the network to GEXF file
    if not os.path.exists(gephi_path):
        os.makedirs(gephi_path)

    gexf_file_path = os.path.join(gephi_path, f"song_network_{year}.gexf")
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

    json_file_path = os.path.join(metrics_path, f"metrics_{year}.json")

    with open(json_file_path, "w") as json_file:
        json.dump(metrics, json_file, indent=4)

    return song_network, metrics


def process_year(
    df, year, features, similarity_threshold, gephi_path, metrics_path
):
    gexf_file_path = os.path.join(gephi_path, f"song_network_{year}.gexf")

    # Check if the file already exists
    if not os.path.exists(gexf_file_path):
        song_network, metrics = generate_network_by_year(
            df, year, features, similarity_threshold, gephi_path, metrics_path
        )
        print(f"Generated network for {year} with {metrics}.")
    else:
        print(f"File for {year} already exists. Skipping.")


class NetworkGenerationThread(threading.Thread):
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
