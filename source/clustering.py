from load import * 
from sklearn.cluster import DBSCAN, Birch, KMeans
import numpy as np
import pandas as pd


final_merge = pd.read_csv('../data/drought-usage'+year+".csv")
keep_fips = final_merge['fips']

# print(keep_fips)
features, labels = read_industry(keep_fips = keep_fips)
# print(features.shape)

model = KMeans(n_clusters=8, n_init = 200, random_state = 1).fit(features)

print(np.unique(model.labels_, return_counts = True))

d = {'fips': list(labels), 'cluster':  model.labels_}
cluster_df = pd.DataFrame(d)
final_merge_cluster = final_merge.merge(cluster_df, left_on = 'fips', right_on='fips')


final_merge_c0 = final_merge_cluster.loc[final_merge_cluster['cluster'] == 3]

indices = ['state_x','county_x','fips', 'none', 'd0', 'd1', 'd2',\
        'd3', 'd4','total_withdrawal_1']
final_merge_small = final_merge_c0[indices]
final_merge_small.loc[final_merge_small['state_x'] =='AL']

idVars = ['state_x','county_x','fips','total_withdrawal_1']

dt_long = pd.melt(final_merge_small, id_vars=idVars, var_name = "level", value_name = 'affected')

dt_long['level'], uniques = pd.factorize(dt_long['level'])

base = 2
dt_long['adjusted_effect'] = (base**dt_long['level'])*dt_long['affected']

# see the data in its final form
# plotting to see data
dt_long.loc[dt_long['level'] == 1]
plt.hist(np.log(1+dt_long['adjusted_effect']), 100)
plt.show()

# plt.hist(np.log(1+dt_long['total_withdrawal_1']), 100)
# plt.show()

plt.scatter(dt_long['adjusted_effect'], np.log(1+dt_long['total_withdrawal_1']))
plt.show()
