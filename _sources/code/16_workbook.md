---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Unsupervised Learning Example - Folktale Clustering

```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```

## Load and Clean the Data


This is the folktales dataset from the Miltilingual Folktales Database

```python
example_text_file = '../data/folktales.csv'

data = pd.read_csv(example_text_file)

data["Index"]=data.index
```

Here we have only decided to do a limited cleaning on the data to assist in our analysis.

```python
# Send to Lowercase
data['Story'] = data['Story'].apply(lambda x: " ".join(x.lower() for x in x.split()))
```

```python
# Remove Punctuation
data['Story'] = data['Story'].str.replace('[^\w\s]','')
```

```python
data['Story'].head()
```

## Using TF-IDF with K-Means Clustering


This takes the TF-IDF matrix and applies a [k-means clustering algorithm](https://en.wikipedia.org/wiki/K-means_clustering). This groups the texts into clusters of similar terms from the TF-IDF matrix. This algorithm randomly seeds X "means", the values are then clustered into the nearest mean. The centroid of the values in each cluster then becomes the new mean and the process repeats until a convergence is reached.

```python
groups = 30

num_clusters = groups
num_seeds = groups
max_iterations = 300
```

```python
pca_num_components = 2
tsne_num_components = 2
```

```python
cmap = matplotlib.cm.get_cmap("nipy_spectral", groups) # Builds a discrete color mapping using a built in matplotlib color map

c = {}
for i in range(cmap.N): # Converts our discrete map into Hex Values
    rgba = cmap(i)
    c.update({i:matplotlib.colors.rgb2hex(rgba)})

labels_color_map=c
```

```python
# calculate tf-idf of texts
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',stop_words= 'english',ngram_range=(1,1))
tf_idf_matrix = tfidf.fit_transform(data['Story'])

# create k-means model with custom config
clustering_model = KMeans(
    n_clusters=num_clusters,
    max_iter=max_iterations#,
    #precompute_distances="auto",
    #n_jobs=-1
)

labels = clustering_model.fit_predict(tf_idf_matrix)
#print(labels)

X = tf_idf_matrix.todense()
```

```python
reduced_data = PCA(n_components=pca_num_components).fit_transform(X)
print(reduced_data)
```

```python
import matplotlib.patches as mpatches
legendlist=[mpatches.Patch(color=labels_color_map[key],label=str(key))for key in labels_color_map.keys()]

fig, ax = plt.subplots()
for index, instance in enumerate(reduced_data):
    #print(instance, index, labels[index])
    pca_comp_1, pca_comp_2 = reduced_data[index]
    color = labels_color_map[labels[index]]
    ax.scatter(pca_comp_1, pca_comp_2, c=color)
if groups<=10:
    plt.legend(handles=legendlist)
plt.show()
```

```python
title_groups = np.transpose([labels,data['Title'],data['Author'],data['Index']])
```

These are the titles of the texts in each cluster. Keep in mind that each time you run the algorithm, the randomness in generating the initial means will result in different clusters.

```python
for i in range(len(title_groups)):
    data.loc[i,'Group'] = title_groups[i][0]
```

This is how the groups break down by number of texts in each.

```python
g_counts = data['Group'].value_counts()
ss = min(g_counts)
#g_counts
```

Here we can look at a sample of the titles in a particular group and the TF-IDF data for that group.

```python
grp=1
data[data['Group']==grp]['Title']#.sample(ss)
```

```python
group_ids = list(data[data['Group']==grp]['Index'])
group_tfidf = data[data['Index'].isin(group_ids)]['Story'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()

group_tfidf.columns = ['words','tf']
group_tfidf.sort_values(by='tf', ascending=False)[:10]

for i,word in enumerate(group_tfidf['words']):
    group_tfidf.loc[i, 'idf'] = np.log(data.shape[0]/(len(data[data['Story'].str.contains(word)])))

group_tfidf['tfidf'] = group_tfidf['tf'] * group_tfidf['idf']
group_tfidf.sort_values(by='tfidf', ascending=False)[:10]
```

```python
text_index = 1
pd.Series(' '.join(data[data['Index']==text_index]['Story']).split()).value_counts()[:10]
```

Here we can look at all the groups along with the authors and the index for each story.

```python
for i in range(groups):
    print("")
    print("#### {} ###".format(i))
    print("")
    for el in title_groups:
        if el[0]==i:
            print("{} --- {} --- #{}".format(el[1],str(el[2]).strip(), el[3]))
```

```python

```
