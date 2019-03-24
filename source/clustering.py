from load import * 
from sklearn.cluster import DBSCAN, Birch, KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler


def exponential(level, affected, base = 2):
    return (base**level)*affected
def linear(level, affected):
    return 2*(1+level)*affected
def polynomial(level, affected):
    return affected**(level)


def extract_cluster_long(final_merge_cluster, cluster_num, withdraw_categories, adjust_fct = None):
    final_merge_c = final_merge_cluster.loc[final_merge_cluster['cluster'] == cluster_num]
    indices = ['state_x','county_x','fips', 'none', 'd0', 'd1', 'd2',\
            'd3', 'd4'] + withdraw_categories
    final_merge_c = final_merge_c[indices]
    idVars = ['state_x','county_x','fips'] + withdraw_categories
    dt_long = pd.melt(final_merge_c, id_vars=idVars, var_name = "level", value_name = 'affected')
    dt_long['level'], uniques = pd.factorize(dt_long['level'])
    if adjust_fct:
        dt_long['adjusted_effect'] = adjust_fct(dt_long['level'], dt_long['affected'])
    else:
        base = 2
        dt_long['adjusted_effect'] = (base**dt_long['level'])*dt_long['affected']
    return dt_long


def plot_usageVeffect(data, baseline, withdraw_category):
    data.dropna(axis=0, inplace = True)
    data = data[(np.abs(stats.zscore(data[withdraw_category])) < 3)]
    plt.scatter(data['adjusted_effect'],  np.log(0.001+ data[withdraw_category]))
    plt.axhline(y= np.log(0.001+baseline))
    plt.show()

withdraw_categories = ['pub_sup_10', 'dom_sup_4', 'ind_7', \
            'irrigation_7', 'crop_3', 'livestock_3', \
            'aqua_7', 'mining_7', 'thermoelectric_27',\
                'gro_wat_1', 'surf_wat_1',
                'total_withdrawal_1']
                
def cluster(final_merge, model, verbose):
    keep_fips = final_merge['fips']
    # print(keep_fips)
    # features, labels, totalPop = read_industry(keep_fips = keep_fips, scale = isScale)
    training_earn, labels_earn = read_earning(keep_fips = keep_fips, scale = isScale)
    model.fit(training_earn)
    if verbose:
        print(np.unique(model.labels_, return_counts = True))
    d = {'fips': list(labels_earn), 'cluster':  model.labels_}
    cluster_df = pd.DataFrame(d)
    final_merge_cluster = final_merge.merge(cluster_df, left_on = 'fips', right_on='fips')
    return final_merge_cluster



def viewEffect(data, withdraw_category):
    baseline = np.mean(data[withdraw_category].loc[data['adjusted_effect'] == 0])
    data_affected = data.loc[data['adjusted_effect'] != 0]
    plot_usageVeffect(data_affected, baseline, withdraw_category)

def regression_usageVeffect(data, withdraw_category, reg_model):
    data_affected = data.loc[data['adjusted_effect'] != 0]
    data_affected.dropna(axis=0, inplace = True)
    X = data_affected['adjusted_effect'].reshape((data_affected['adjusted_effect'].shape[0], 1))
    y = data_affected[withdraw_category]
    scaler = StandardScaler()
    X_new = scaler.fit_transform(X)
    y_new = scaler.fit_transform(y)
    reg_model.fit(X_new, y_new)
    print(reg_model.coef_)
    return reg_model




# merge_water(withdraw_categories)
final_merge = pd.read_csv('../data/drought-usage'+str(year)+".csv")
model = KMeans(n_clusters=5, n_init = 200, random_state = 1234)
final_merge_cluster = cluster(final_merge, model, verbose)

data = extract_cluster_long(final_merge_cluster, 1, withdraw_categories, linear)

withdraw_category = 'total_withdrawal_1'

reg = linear_model.LinearRegression()
regression_usageVeffect(data, withdraw_category, reg_model)