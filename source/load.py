import pandas as pd
import numpy as np 
import datetime
import matplotlib.pyplot as plt

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
    levels = ['d0', 'd1', 'd2', 'd3', 'd4']
    final_drought_dt['level_d'] = np.zeros(final_drought_dt.shape[0])
    for i, level in enumerate(levels):
        per10Ind = final_drought_dt[level]>10.0
        final_drought_dt['level_d'].loc[per10Ind] = i*np.ones(sum(per10Ind))
    final_drought_dt.to_csv('../data/final_drought'+str(year)+".csv")




def merge_water(withdraw_categories):
    drought_dt = pd.read_csv('../data/final_drought'+str(year)+".csv", encoding= "latin-1")
    water = pd.read_csv("../data/water_usage.csv")
    indices = ['state','county','fips', 'population'] + withdraw_categories
    water['population'] = water['population'].convert_objects(convert_numeric=True)
    water[withdraw_categories] = water[withdraw_categories].convert_objects(convert_numeric=True)
    water[withdraw_categories] = water[withdraw_categories].div(water['population'], axis = 0)
    water = water[indices]
    final_merge = drought_dt.merge(water,  left_on = 'fips', right_on='fips')
    drought_levels = ['none', 'd0', 'd1', 'd2','d3', 'd4']
    final_merge[drought_levels] = final_merge[drought_levels].mul(final_merge['population'], axis = 0)
    final_indices = ['state_x','county_x','fips', 'none', 'd0', 'd1', 'd2',
        'd3', 'd4', 'level_d'] + withdraw_categories
    final_merge = final_merge[final_indices]
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
    return training, labels

