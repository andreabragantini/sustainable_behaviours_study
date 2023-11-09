import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
from sklearn.manifold import TSNE
import matplotlib.cm as cm


# Read the data
target = "sprener"
#target = "spracqua"
y = pd.read_csv("data/tar_{}.csv".format(target))
X = pd.read_csv("data/preprocessed_data.csv")


#%% Visualization with tSNE
'''t-distributed stochastic neighbor embedding (t-SNE) is a statistical method for visualizing high-dimensional data 
by giving each datapoint a location in a two or three-dimensional map.'''
# t-SNE reduces dimensionality into 2 dimensions, so we can plot and visualize data set.

print("\nApplying t-SNE...")
start_tsne = time.time()
# instantiate a TSNE object
tsne = TSNE()
# Apply t-SNE to reduce the dimensionality of the features to 2D
X_tsne = TSNE().fit_transform(X)
X_tsne

# coordinates to plot in 2 dimensions
vis_x = X_tsne[:, 0]
vis_y = X_tsne[:, 1]

# Create a scatter plot of the t-SNE representation of the data, with different colors for each target value
cmap = cm.get_cmap('viridis')       # choose a colormap to use
plt.figure(figsize=(10, 10), dpi=300)
plt.scatter(vis_x, vis_y, c=y, s=5, cmap=cmap)
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.title('t-SNE visualization for {}'.format(y.name))
plt.colorbar()
plt.savefig('exploratory_analysis/tsne_{}.png'.format(y.name))

# save the TSNE model to a file.
with open("exploratory_analysis/tsne.pkl", "wb") as f:
    pickle.dump(tsne, f)

end_tsne = time.time()
print("t-SNE completed in: {:.2f} minutes".format((end_tsne - start_tsne) / 60))



#%% CLUSTERING
from sklearn.cluster import KMeans
num_clusters = 18
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Add cluster labels to the original dataset
X['cluster_label'] = kmeans.labels_
cluster_labels = kmeans.labels_
# Now, 'X' contains an additional column 'cluster_label' indicating the cluster assignment for each sample

# To see the cluster assignments and inspect the clusters:
print(X['cluster_label'].value_counts())


# You can further analyze the clusters to identify groups of variables that are redundant
# For example, you can calculate the mean/variance of variables within each cluster and compare them
# to see if there are clusters with similar mean/variance values
# You can also plot the clusters to see if there are any patterns
# For example, you can plot the first two principal components of the data and color the samples by cluster label
# to see if the clusters are well-separated
# You can also plot the clusters using t-SNE or UMAP to see if there are any patterns

# visualization of the clustering results
# Create a scatter plot of the t-SNE representation of the data, with different colors for each cluster
plt.figure(figsize=(8, 8), dpi=300)
for i in range(len(np.unique(cluster_labels))):
    plt.scatter(X_tsne[cluster_labels == i, 0], X_tsne[cluster_labels == i, 1], label=f'Cluster {i}')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.title('KMeans Clustering Results (t-SNE)')
plt.legend()
plt.savefig('exploratory_analysis/tsne_clustering_{}.png'.format(y.name))