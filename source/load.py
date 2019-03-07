# earnings 
# doughts 
# educational attainment 

import pandas as pd
import numpy as np 
import datetime
import matplotlib.pyplot as plt


# earnings = pd.read_csv("earnings.csv", encoding= "latin-1")
# edu = pd.read_csv("education_attainment.csv", encoding= "latin-1")
# indu = pd.read_csv("industry_occupation.csv", encoding= "latin-1")

year = '2010'
def parse_drought():
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    droughts = pd.read_csv("../data/droughts.csv", parse_dates=['valid_start', "valid_end"], date_parser=dateparse, encoding= "latin-1")
    s_droughts = droughts.loc[(droughts['d0']>0.0) | \
                    (droughts['d1']>0.0) | \
                    (droughts['d2']>0.0)|\
                    (droughts['d3']>0.0)|\
                    (droughts['d4']>0.0)
                    ]
    s_droughts = s_droughts.loc[(s_droughts['valid_start'] >= year+'-01-01') & \
        (s_droughts['valid_end'] <= year+'-12-31')]
    countyState = s_droughts[s_droughts.columns[0:3]].drop_duplicates()
    sum_affected = s_droughts.groupby('fips').mean()
    sum_affected['fips'] = sum_affected.index
    final_drought_dt = countyState.merge(sum_affected, left_on = 'fips', right_on='fips')

    final_drought_dt.to_csv('../data/final_drought'+year+".csv")

def merge_water():
    drought_dt = pd.read_csv('../data/final_drought'+year+".csv", encoding= "latin-1")
    water = pd.read_csv("../data/water_usage.csv")

    indices = ['state','county','fips', 'pub_sup_12', 'dom_sup_8', 'ind_9', \
            'irrigation_7', 'crop_7', 'livestock_3', \
            'aqua_9', 'mining_9', 'thermoelectric_29',\
                'total_withdrawal_1','total_withdrawal_3']
    water = water[indices]
    final_merge = drought_dt.merge(water,  left_on = 'fips', right_on='fips')
    final_indices = ['state_x','county_x','fips', 'none', 'd0', 'd1', 'd2',
        'd3', 'd4', 'pub_sup_12', 'dom_sup_8', 'ind_9', \
            'irrigation_7', 'crop_7', 'livestock_3', \
            'aqua_9', 'mining_9', 'thermoelectric_29',\
                'total_withdrawal_1','total_withdrawal_3']
    final_merge = final_merge[final_indices]
    final_merge.to_csv('../data/drought-usage'+year+".csv")

# uncomment to make the merged drought-usage2010.csv in your directory (only have to run once)
# parse_drought()
# merge_water()
final_merge = pd.read_csv('../data/drought-usage'+year+".csv")
indices = ['state_x','county_x','fips', 'none', 'd0', 'd1', 'd2',\
        'd3', 'd4','total_withdrawal_1']
final_merge_small = final_merge[indices]
final_merge_small.loc[final_merge_small['state_x'] =='AL']

idVars = ['state_x','county_x','fips','total_withdrawal_1']

dt_long = pd.melt(final_merge_small, id_vars=idVars, var_name = "level", value_name = 'affected')

dt_long['level'], uniques = pd.factorize(dt_long['level'])

base = 2
dt_long['adjusted_effect'] = (base**dt_long['level'])*dt_long['affected']

# see the data in its final form
print(dt_long.head())
# plotting to see data
# dt_long.loc[dt_long['level'] == 1]
# plt.hist(np.log(1+dt_long['adjusted_effect']), 100)
# plt.show()

# plt.hist(np.log(1+dt_long['total_withdrawal_1']), 100)
# plt.show()

# plt.scatter(dt_long['adjusted_effect'], np.log10(1+dt_long['total_withdrawal_1']))
# plt.show()