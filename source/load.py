# earnings 
# doughts 
# educational attainment 

import pandas as pd
import numpy as np 
import datetime
import matplotlib.pyplot as plt


# edu = pd.read_csv("education_attainment.csv", encoding= "latin-1")
# indu = pd.read_csv("industry_occupation.csv", encoding= "latin-1")

year = 2010
def parse_drought():
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    droughts = pd.read_csv("../data/droughts.csv", parse_dates=['valid_start', "valid_end"], date_parser=dateparse, encoding= "latin-1")
    s_droughts = droughts.loc[(droughts['d0']>0.0) | \
                    (droughts['d1']>0.0) | \
                    (droughts['d2']>0.0)|\
                    (droughts['d3']>0.0)|\
                    (droughts['d4']>0.0)
                    ]
    s_droughts = s_droughts.loc[(s_droughts['valid_start'] >= str(year)+'-01-01') & \
        (s_droughts['valid_end'] <= str(year)+'-12-31')]
    countyState = s_droughts[s_droughts.columns[0:3]].drop_duplicates()
    sum_affected = s_droughts.groupby('fips').mean()
    sum_affected['fips'] = sum_affected.index
    final_drought_dt = countyState.merge(sum_affected, left_on = 'fips', right_on='fips')

    final_drought_dt.to_csv('../data/final_drought'+str(year)+".csv")



def merge_water(withdraw_catergories):
    drought_dt = pd.read_csv('../data/final_drought'+str(year)+".csv", encoding= "latin-1")
    water = pd.read_csv("../data/water_usage.csv")
    indices = ['state','county','fips', 'population'] + withdraw_catergories
    water['population'] = water['population'].convert_objects(convert_numeric=True)
    water[withdraw_catergories] = water[withdraw_catergories].convert_objects(convert_numeric=True)
    water[withdraw_catergories] = water[withdraw_catergories].div(water['population'], axis = 0)
    water = water[indices]
    final_merge = drought_dt.merge(water,  left_on = 'fips', right_on='fips')
    drought_levels = ['none', 'd0', 'd1', 'd2','d3', 'd4']
    final_merge[drought_levels] = final_merge[drought_levels].mul(final_merge['population'], axis = 0)
    final_indices = ['state_x','county_x','fips', 'none', 'd0', 'd1', 'd2',
        'd3', 'd4'] + withdraw_catergories
    final_merge = final_merge[final_indices]
    print(final_merge.head(100))
    final_merge.to_csv('../data/drought-usage'+str(year)+".csv")


def read_industry(keep_fips, scale = False):
    indu = pd.read_csv('../data/industry_occupation.csv', encoding= "latin-1")
    indu.dropna(axis=0, inplace = True)
    # indu_fips = set(indu['fips'])
    indu = indu.groupby('fips').mean()
    indu['fips'] = indu.index
    indu = indu.loc[indu['fips'].isin(keep_fips)]
    features = ['agriculture',  'construction',  'manufacturing' ,\
         'wholesale_trade', 'retail_trade', \
             'transport_utilities','information','finance_insurance_realestate'
                ]
    training = indu[features]
    if scale:
        training.div(indu['total_employed'], axis = 0)
    labels = indu['fips']
    return training, labels, indu['total_employed']



def read_earning(keep_fips, scale = False):
    earnings = pd.read_csv("../data/earnings.csv", encoding= "latin-1")
    earnings2010 = earnings.loc[earnings['year'] == year]
    earnings2010 = earnings2010.loc[earnings2010['fips'].isin(keep_fips)]
    features_earn = ['fips', 'total_agri_fish_mine', \
        'construction', 'manufacturing', 'wholesale_trade', 'retail_trade',\
                'utilities', 'information',\
                    'fin_ins_realest', \
                        'total_prof_sci_mgmt_admin', \
                            'total_edu_health_social', \
                                'total_arts_ent_acc_food',\
                                        'pub_admin']
    training = earnings2010[features_earn]
    training = training.convert_objects(convert_numeric=True)
    training.dropna(axis=0, inplace = True)
    labels = training['fips']
    training = training[features_earn[1:]]
    training.dropna(axis=0, inplace = True)
    # if scale:
    #     total_med = earnings2010['total_med']
    #     total_med = total_med.convert_objects(convert_numeric=True)
    #     training = training.div(total_med,  axis = 0)
    return training, labels


# uncomment to make the merged drought-usage2010.csv in your directory (only have to run once)
# parse_drought()
# # merge_water()
# final_merge = pd.read_csv('../data/drought-usage'+year+".csv")
# indices = ['state_x','county_x','fips', 'none', 'd0', 'd1', 'd2',\
#         'd3', 'd4','total_withdrawal_1']
# final_merge_small = final_merge[indices]
# final_merge_small.loc[final_merge_small['state_x'] =='AL']

# idVars = ['state_x','county_x','fips','total_withdrawal_1']

# dt_long = pd.melt(final_merge_small, id_vars=idVars, var_name = "level", value_name = 'affected')

# dt_long['level'], uniques = pd.factorize(dt_long['level'])

# base = 2
# dt_long['adjusted_effect'] = (base**dt_long['level'])*dt_long['affected']

# # see the data in its final form
# print(dt_long.head())
# # plotting to see data
# dt_long.loc[dt_long['level'] == 1]
# plt.hist(np.log(1+dt_long['adjusted_effect']), 100)
# plt.show()

# plt.hist(np.log(1+dt_long['total_withdrawal_1']), 100)
# plt.show()

# plt.scatter(dt_long['adjusted_effect'], np.log10(1+dt_long['total_withdrawal_1']))
# plt.show()


