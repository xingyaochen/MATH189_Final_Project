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


def extract_cluster(final_merge_cluster, cluster_num, withdraw_categories):
    final_merge_c = final_merge_cluster.loc[final_merge_cluster['cluster'] == cluster_num]
    indices = ['state_x','county_x','fips', 'level_d'] + withdraw_categories
    final_merge_c = final_merge_c[indices]
    return final_merge_c


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
    plt.scatter(data['adjusted_effect'],  np.log(data[withdraw_category]))
    plt.axhline(y= np.log(baseline))
    plt.savefig("plots/"+withdraw_category+"Veffect.png")
    plt.close()


def plot_usageVeffect_d(data, withdraw_category, cluster, means = None, log = True):
    data.dropna(axis=0, inplace = True)
    data = data[(np.abs(stats.zscore(data[withdraw_category])) < 3)]
    if log:
        data_nonzero = data.loc[ data[withdraw_category] > 0]
        y = np.log(data_nonzero[withdraw_category]) 
        X = data_nonzero['level_d']
    else:
        y = data[withdraw_category]
        X = data['level_d']
    plt.scatter(X, y)
    if means.size > 0:
        plt.plot(means['level_d'], means[withdraw_category], 'r^-')
    plt.savefig("plots/"+"c" + str(cluster)+"_"+ withdraw_category + "Veffect_d.png")
    plt.close()



                
def cluster_by_earning(final_merge, model, verbose = False):
    keep_fips = final_merge['fips']
    # print(keep_fips)
    # features, labels, totalPop = read_industry(keep_fips = keep_fips, scale = isScale)
    training_earn, labels_earn = read_earning(keep_fips = keep_fips)
    model.fit(training_earn)
    if verbose:
        print(np.unique(model.labels_, return_counts = True))
    d = {'fips': list(labels_earn), 'cluster':  model.labels_}
    cluster_df = pd.DataFrame(d)
    final_merge_cluster = final_merge.merge(cluster_df, left_on = 'fips', right_on='fips')
    return final_merge_cluster


def cluster_by_usage(final_merge, withdraw_categories,  model, verbose = False):
    training = final_merge[['fips'] + withdraw_categories]
    training.convert_objects(convert_numeric=True)
    training.dropna(axis=0, inplace = True)
    labels = training['fips']
    training = training[withdraw_categories]
    model.fit(training)
    if verbose:
        print(np.unique(model.labels_, return_counts = True))
    d = {'fips': list(labels), 'cluster':  model.labels_}
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
    y = np.log(data_affected[withdraw_category])
    reg_model.fit(X, y)
    print(withdraw_category)
    print("Score:" , reg_model.score(X, y))
    print("Coef:" , reg_model.coef_)
    print("Intercept:", reg_model.intercept_)
    return reg_model


def regression_usageVeffect_d(data, withdraw_category, reg_model, plot = False, log = True):
    data.dropna(axis=0, inplace = True)
    data = data[(np.abs(stats.zscore(data[withdraw_category])) < 3)]
    if log:
        data_nonzero = data.loc[ data[withdraw_category] > 0]
        y = np.log(data_nonzero[withdraw_category]) 
        X = data_nonzero['level_d']
    else:
        y = data[withdraw_category]
        X = data['level_d']
    X = X.reshape((X.shape[0], 1))
    reg_model.fit(X, y)
    if plot:
        plot_usageVeffect_d(data, withdraw_category, log)
    print(withdraw_category)
    print("Score:" , reg_model.score(X, y))
    print("Coef:" , reg_model.coef_)
    print("Intercept:", reg_model.intercept_)
    print('\n')
    return reg_model




def get_mean_effect(data, withdraw_category, cluster, plot = False, log = False):
    # data = data[(np.abs(stats.zscore(data[withdraw_category])) < 3)]
    if log:
        data_nonzero = data.loc[ data[withdraw_category] > 0]
        X = data_nonzero['level_d']
        y = np.log(data_nonzero[withdraw_category]) 
        data_nonzero[withdraw_category] = y
        withdraw_means = data_nonzero[['level_d', withdraw_category]].groupby('level_d').mean()
    else:
        X = data['level_d']
        y = data[withdraw_category]
        withdraw_means = data[['level_d', withdraw_category]].groupby('level_d').mean()
    # print(withdraw_means)
    withdraw_means.to_csv('c'+str(cluster)+"_"+ withdraw_category + ".csv", sep='\t', encoding='utf-8')

    withdraw_means['level_d'] = withdraw_means.index
    if plot:
        plot_usageVeffect_d(data, withdraw_category, cluster,withdraw_means, log)
    return withdraw_means

def get_sum_slope(withdraw_means, withdraw_category):
    means = list(withdraw_means[withdraw_category])
    return sum([j-i for i, j in zip(means[:-1], t[1:])])


withdraw_categories = ['pub_sup_10', 'dom_sup_4', 'ind_7', \
            'irrigation_7', 'crop_3', 'livestock_3', \
            'aqua_7', 'mining_7', 'thermoelectric_27',\
                'gro_wat_1', 'surf_wat_1',
                'total_withdrawal_1']

merge_water(withdraw_categories)
final_merge = pd.read_csv('../data/drought-usage'+str(year)+".csv")
model = KMeans(n_clusters=5, n_init = 200, random_state = 1234)
# final_merge_cluster = cluster_by_usage(final_merge, withdraw_categories,  model, verbose = True)

final_merge_cluster = cluster_by_earning(final_merge, model, verbose=True)

clusters = np.unique(model.labels_)
for c in clusters:
    data = extract_cluster(final_merge_cluster, c, withdraw_categories)
    for withdraw_category in withdraw_categories:
        print("cluster", c)
        print('withdraw category', withdraw_category)
        withdraw_means = get_mean_effect(data, withdraw_category, c, plot = True, log = True)
        slope_sum = get_sum_slope(withdraw_means, withdraw_category)