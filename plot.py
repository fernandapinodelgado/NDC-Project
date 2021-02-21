import os
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

vectors = pd.read_csv('csv/vectors.csv')
shapefile = os.path.expanduser('ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp')

gdf = gpd.read_file(shapefile)[['ADM0_A3', 'geometry']].to_crs('+proj=robin')

df = vectors[['ID', 'environment']]
merged = gdf.merge(df, left_on='ADM0_A3', right_on='ID')

print(merged.sample(5))
print(merged.describe())

colors = 9
cmap = 'Blues'
figsize = (16, 10)

ax = merged.plot(cmap=cmap, figsize=figsize, scheme='equal_interval', k=colors, legend=True)
plt.show()

