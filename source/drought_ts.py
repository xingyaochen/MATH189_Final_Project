from load import * 
from pandas import Series 
import matplotlib.pyplot as plt
from county_adjacency import * 
from dtaidistance import dtw
import time 


def find_max_dist_counties(droughts, max_threshold):
    min_threshold = 1
    # max_threshold = 15
    max_distMat_L = []
    candidate_groupL = set()
    test_min_groupL = set()
    adj =  make_county_adjacencyL()
    for group in list(adj.values()):
        # print(group)
        if len(group) > 1:
            drought_subset = droughts.loc[\
            (droughts['fips'].isin(group))]
            series = []
            if len(np.unique(drought_subset['fips'])) > 1:
                for key, grp in drought_subset.groupby(['fips']):
                    # ax = grp.plot(ax=ax, kind='line', x='valid_start', y='level_d', label=key)
                    ts_i = np.array(grp['level_d'].tolist())
                    series.append(ts_i)
                ds = dtw.distance_matrix_fast(series)
                time.sleep(0.01)
                min_dists = sorted(np.amin(ds, axis = 1))
                
                if min_dists[1]  < min_threshold:
                    print("min", min_dists)
                    print(group)
                    test_min_groupL.add(tuple(sorted(np.unique(drought_subset['fips']))))
                elif min_dists[-2]  > max_threshold:
                    candidate_groupL.add(tuple(sorted(np.unique(drought_subset['fips']))))
                    max_distMat_L.append(ds)
                    print("max", min_dists)
                    print(group)
                else:
                    continue
    return candidate_groupL, test_min_groupL

def plot_groups_ts(candidate_groupL, save_weekly = False):
    for i, c_group in enumerate(list(candidate_groupL)):
        drought_subset = droughts.loc[(droughts['fips'].isin(c_group))]
        if save_weekly:
            drought_subset.to_csv('../data/drought_ts_data/drought_subset_group'+ str(i).zfill(2) + ".csv")
        fig, ax = plt.subplots()
        for key, grp in drought_subset.groupby(['fips']):
            lab = grp['county'].iloc[0] +", "+grp['state'].iloc[0]
            ax = grp.plot(ax=ax, kind='line', x='valid_start', y='level_d', label=lab, linewidth = 1)
        plt.legend(loc='best', fontsize = 7)
        plt.title("Drought Levels | 2010 - 2017")
        plt.savefig('drought_ts_plots/min_dist_group'+str(i).zfill(2)+".png")
        plt.close()


droughts = parse_drought(save_weekly = False)
candidate_groupL, test_min_groupL = find_max_dist_counties(droughts, 13)
plot_groups_ts(candidate_groupL, save_weekly = False)
parse_drought_subset('drought_ts_data')