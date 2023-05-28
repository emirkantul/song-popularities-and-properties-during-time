# song-popularities-and-properties-during-time
Analyzing Song Popularity and Properties During Time with Network Science

## For VSCODE
- Install poetry https://python-poetry.org/docs/
- run `poetry install` on main folder (folder containing this README.md)
- Now you can work with NetworkAnalysis.ipynb 
### Domain of Interest:
The domain of interest for this project is the analysis of music data, specifically using network science to understand the relationships between songs based on their musical properties, and how these relationships have changed over time.

### Data Source:
The data for this project will be obtained from Spotify's API, which provides access to a large collection of songs along with their corresponding musical properties such as danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, and valence. In addition to the musical properties, we will also use the popularity of the songs as a measure of their success.

### Main Hypotheses:
The main hypotheses for this project are twofold:
- Songs that are similar in their musical properties are likely to be related to each other in a network.
- The relationships between songs and their popularity have changed over time, and can be revealed through network analysis.

### Current Literature:
The current literature on music analysis and network science suggests that there is a strong relationship between musical properties and the structure of musical networks. Research has shown that songs with similar musical properties tend to cluster together in networks, and that these clusters can be used to identify musical genres and sub-genres. However, there is still much to be explored in this area, particularly with regard to the use of network analysis techniques to gain insights into the structure and properties of musical networks over time.

### Planned Analysis:
The planned analysis for this project involves creating a network of songs using their musical properties as described above. We will then use various network analysis techniques to gain insights into the structure and properties of the resulting network, year by year. Specifically, we plan to use community detection algorithms to identify clusters of similar songs, and centrality measures to identify the most influential songs in the network. We will also examine how the network changes and grows year by year, and how the relationships between song properties and popularity have evolved over time.

### Relevance to Network Science:
This research is highly relevant to network science, as it involves the analysis of a network of songs based on their properties over time. By using network analysis techniques, we hope to gain insights into the structure and properties of the resulting network, and to uncover relationships between songs and their popularity that may not be immediately apparent through other means.

### Methodologies:
The methodologies we plan to use for this project include data normalization, similarity measures, graph theory, and network analysis techniques such as community detection and centrality measures. We will use Python programming language along with popular data science and network analysis libraries such as Pandas, NumPy, NetworkX, and Plotly to carry out the analysis. We will also use time series analysis techniques to examine how the network changes and evolves over time.

### References:
- Prediction of product success: Explaining song popularity by audio ... (n.d.). RetrievedApril 3, 2023, from https://essay.utwente.nl/75422/1/NIJKAMP_BA_IBA.pdf 
- Musical influence network analysis and rank of sample-based music. (n.d.). RetrievedApril 3, 2023, from https://www.researchgate.net/publication/220723768_Musical_Influence_Network_Analysis_and_Rank_of_Sample-Based_Music
- Mavani, V. (2021, December 17). Spotify dataset. Kaggle. Retrieved April 3, 2023, fromhttps://www.kaggle.com/datasets/vatsalmavani/spotify-dataset
