# Player Clustering Project

## Overview

This project clusters player data based on features such as overall rating, potential, value in euros, wage in euros, and age using the K-means algorithm.

## Features

- `overall`
- `potential`
- `value_eur`
- `wage_eur`
- `age`

## Steps

1. **Data Preprocessing**: 
   - Drop rows with missing values in the selected features.
   - Normalize the data.

    ```python
    players = players.dropna(subset=features)
    data = ((players[features] - players[features].min()) / (players[features].max() - players[features].min())) * 10 + 1
    ```

2. **Initialize Random Centroids**:

    ```python
    def random_centroids(data, k):
        return pd.concat([data.apply(lambda x: float(x.sample())) for _ in range(k)], axis=1)
    centroids = random_centroids(data, 5)
    ```

3. **Assign Labels**:

    ```python
    def get_labels(data, centroids):
        distances = centroids.apply(lambda x: np.sqrt(((data - x) ** 2).sum(axis=1)))
        return distances.idxmin(axis=1)
    labels = get_labels(data, centroids)
    ```

4. **Update Centroids**:

    ```python
    def new_centroids(data, labels):
        return data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T
    ```

5. **Visualize Clusters**:

    ```python
    def plot_clusters(data, labels, centroids, iteration):
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        centroids_2d = pca.transform(centroids.T)
        clear_output(wait=True)
        plt.title(f'Iteration {iteration}')
        plt.scatter(data_2d[:,0], data_2d[:,1], c=labels)
        plt.scatter(centroids_2d[:,0], centroids_2d[:,1])
        plt.show()
    ```

6. **Iterative Optimization**:

    ```python
    max_iterations, centroid_count = 200, 3
    centroids = random_centroids(data, centroid_count)
    old_centroids = pd.DataFrame()
    iteration = 1

    while iteration < max_iterations and not centroids.equals(old_centroids):
        old_centroids = centroids
        labels = get_labels(data, centroids)
        centroids = new_centroids(data, labels)
        plot_clusters(data, labels, centroids, iteration)
        iteration += 1
    ```

7. **Inspect Results**:

    ```python
    players[labels==0][["short_name"] + features]
    ```

This process clusters players to identify groups with similar characteristics.



You can find the dataset here! https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset
