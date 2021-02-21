import pandas as pd  # for data tools
from sklearn.cluster import KMeans  # for clustering tools
import csv  # for data output formatting

tags = ['mitigation',
        'adaptation',
        'action',
        'information',
        'objective',
        'emissions',
        'sector',
        'planning',
        'policy',
        'economic',
        'environment',
        'context',
        'ambition',
        'vulnerability',
        'institutions',
        'projection',
        'agriculture',
        'needs',
        'transparency',
        'equity',
        'development',
        'energy',
        'forestry',
        'population',
        'waste',
        'reduction',
        'achievement',
        'industry',
        'technology',
        'reporting',
        'sustainable',
        'inventory',
        'marine',
        'education',
        'health']

tfidf_data = pd.read_csv('csv/tfidf.csv')
tfidf_data.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
# print(tfidf_data)

vectors = tfidf_data[tags]
vectors_labeled = tfidf_data[tags+['ID']].set_index('ID')
vectors_labeled.to_csv('csv/vectors.csv')  # Only run to output whole dataset
# print(vectors)

kmeans = KMeans().fit(vectors)
print(kmeans.labels_)

labels = {}
for i in range(len(vectors)):
    country = tfidf_data.iloc[i]['ID']
    label = kmeans.labels_[i]

    if label in labels:
        labels[label].append(country)
    else:
        labels[label] = [country]

print(labels.keys())

with open('clusters.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in labels.items():
        writer.writerow([key, value])

'''
Notes: (last updated 02/15/2021)
- Find a better number of clusters than 8
- Or maybe just a whole other clustering alg
- Also find some way to meaningfully distinguish the clusters that get made
'''
