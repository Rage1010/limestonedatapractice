# Import necessary libraries
from sklearn import datasets # to retrieve the iris Dataset
import pandas as pd # to load the dataframe
from sklearn.preprocessing import StandardScaler # to standardize the features
from sklearn.decomposition import PCA # to apply PCA
from sklearn.cluster import KMeans
import seaborn as sns # to plot the heat maps
import matplotlib.pyplot as plt

df = pd.read_csv("returns.csv")
df = df.transpose()
# print(df.head())

#Standardize the features
#Create an object of StandardScaler which is present in sklearn.preprocessing
scalar = StandardScaler() 
scaled_data = pd.DataFrame(scalar.fit_transform(df)) #scaling the data
# print(scaled_data)

# sns.heatmap(scaled_data.corr())

#Applying PCA
#Taking no. of Principal Components as {n_components}

n_components = 3
pca = PCA(n_components = n_components)
pca_columns = [f'PC{i}' for i in range(1,n_components+1)]
pca.fit(scaled_data)
data_pca = pca.transform(scaled_data)
data_pca = pd.DataFrame(data_pca,columns=pca_columns)
print(data_pca.head())
heat_map = sns.heatmap(data_pca.corr())
plt.show()


#initialize kmeans parameters
kmeans_kwargs = {
"init": "random",
"n_init": 10,
"random_state": 1,
}

#create list to hold SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(data_pca)
    sse.append(kmeans.inertia_)

#visualize results
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

n_clusters = 4
#instantiate the k-means class, using optimal number of clusters
kmeans = KMeans(init="random", n_clusters=n_clusters, n_init=10, random_state=1)

#fit k-means algorithm to data
kmeans.fit(data_pca)
df['cluster'] = kmeans.labels_
# print(df)
# print(data_pca)

# inertia_values = []
# inertia_values.append(kmeans.inertia_)
# y_kmeans = kmeans.fit_predict(data_pca)
# inertia_values.append(kmeans.inertia_)
# plt.scatter(data_pca.iloc[:, 'PC1'], data_pca.iloc[:, 'PC2'], c=y_kmeans)
# plt.scatter(kmeans.cluster_centers_[:, 0],\
#                 kmeans.cluster_centers_[:, 1], \
#                 s=100, c='red')
# plt.title('K-means clustering (k={})'.format(n_clusters))
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.show()



lulu = [[] for i in range(n_clusters)]
for ind in df.index:
    lulu[int(df["cluster"][ind])].append(ind)
for i in range(n_clusters):
    print(lulu[i])
    print(len(lulu[i]))
