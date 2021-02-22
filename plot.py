import os
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

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

EU_codes = ['BEL', 'BGR', 'HRV', 'CZE', 'DNK', 'DEU', 'EST', 'IRL', 'GRC', 'ESP', 'FRA', 'ITA', 'CYP', 'LVA', 'LTU',
            'LUX', 'HUN', 'MLT', 'NLD', 'AUT', 'POL', 'PRT', 'ROU', 'SVN', 'SVK', 'FIN', 'SWE', 'GBR']

vectors = pd.read_csv('csv/vectors.csv')
shapefile = os.path.expanduser('ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp')

gdf = gpd.read_file(shapefile)[['ADM0_A3', 'geometry']].to_crs('+proj=robin')

for query in tags:
    df = vectors[['ID', query]]

    # Add EU countries to df
    val = df.loc[df['ID'] == 'EU28']
    val = val.at[val.index[0], query]
    # print(val)
    for code in EU_codes:
        df = df.append({'ID': code, query: val}, ignore_index=True)

    merged = gdf.merge(df, left_on='ADM0_A3', right_on='ID')

    print(merged.sample(5))
    print(merged.describe())

    colors = 9
    cmap = 'Blues'
    figsize = (16, 10)

    ax = merged.plot(column=query, cmap=cmap, figsize=figsize, scheme='equal_interval', k=colors, legend=True)
    merged[merged.isna().any(axis=1)].plot(ax=ax, color='#fafafa', hatch='///')
    ax.set_title('TF-IDF scores for \"' + query + '\" in NDCs')
    ax.annotate('TF-IDF analysis performed on various countries\' NDCs/INDCs, distribution for the word \"' + query + '\".',
                xy=(0.1, 0.1), size=12, xycoords='figure fraction')
    ax.set_axis_off()
    ax.set_xlim([-1.5e7, 1.7e7])
    ax.get_legend().set_bbox_to_anchor((.12, .4))
    plt.savefig('images/' + query + '.png')
    plt.close()
