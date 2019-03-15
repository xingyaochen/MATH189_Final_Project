from load import * 
from sklearn.cluster import DBSCAN, Birch, KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


final_merge = pd.read_csv('../data/drought-usage'+year+".csv")
keep_fips = final_merge['fips']

# print(keep_fips)
features, labels = read_industry(keep_fips = keep_fips)
# print(features.shape)

model = KMeans(n_clusters=8, n_init = 200, random_state = 1234).fit(features)

print(np.unique(model.labels_, return_counts = True))

d = {'fips': list(labels), 'cluster':  model.labels_}
cluster_df = pd.DataFrame(d)
final_merge_cluster = final_merge.merge(cluster_df, left_on = 'fips', right_on='fips')



def extract_cluster_long(final_merge_cluster, cluster_num, adjust_fct = None):
    final_merge_c = final_merge_cluster.loc[final_merge_cluster['cluster'] == cluster_num]
    print(final_merge_c.shape)
    indices = ['state_x','county_x','fips', 'none', 'd0', 'd1', 'd2',\
            'd3', 'd4','total_withdrawal_1']
    final_merge_c = final_merge_c[indices]
    idVars = ['state_x','county_x','fips','total_withdrawal_1']
    dt_long = pd.melt(final_merge_c, id_vars=idVars, var_name = "level", value_name = 'affected')
    dt_long['level'], uniques = pd.factorize(dt_long['level'])
    base = 2
    dt_long['adjusted_effect'] = (base**dt_long['level'])*dt_long['affected']
    return dt_long


def plot_usageVeffect(data, baseline):
    # print(data.head())
    plt.scatter(data['affected'], np.log(1+data['total_withdrawal_1']))
    plt.axhline(y=np.log(1+baseline))
    plt.show()


data = extract_cluster_long(final_merge_cluster, 0)
baseline = np.mean(data['total_withdrawal_1'].loc[data['level'] == 0])
data_affected = data.loc[data['level'] != 0]
plot_usageVeffect(data_affected, baseline)