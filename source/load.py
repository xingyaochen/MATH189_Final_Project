import pandas as pd
import numpy as np 
import datetime
import matplotlib.pyplot as plt

year = 2010
def parse_drought(save_weekly = False):
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    droughts = pd.read_csv("../data/droughts.csv",  parse_dates = ['valid_start', 'valid_end'], date_parser = dateparse, encoding= "latin-1")
    droughts = droughts.loc[(droughts['d0']>0.0) | \
                    (droughts['d1']>0.0) | \
                    (droughts['d2']>0.0)|\
                    (droughts['d3']>0.0)|\
                    (droughts['d4']>0.0)
                    ]
    print(droughts.shape)
    all_weeks = np.unique(droughts['valid_start'])
    droughts['level_d'] = np.zeros(droughts.shape[0])
    levels = ['d0', 'd1', 'd2', 'd3', 'd4']
    for i, level in enumerate(levels):
        per10Ind = droughts[level]>10.0
        droughts['level_d'].loc[per10Ind] = i*np.ones(sum(per10Ind))
    if save_weekly:
        for i, wk in enumerate(all_weeks):
            weekly_drought = droughts.loc[droughts['valid_start'] == wk]
            weekly_drought.to_csv('../data/weekly_drought_'+str(wk)+".csv")
            print(i)
    return droughts

def parseEdu():
    edu = pd.read_csv("../data/education_attainment.csv", encoding= "latin-1")
    percFeatures = ['fips', 'year','pct_less_than_hs', 'pct_hs_diploma', 'pct_college_or_associates', 'pct_college_bachelors_or_higher']
    edu = edu[percFeatures]
    edu_latest = edu.loc[edu['year'] == '2012-2016']
    edu_latest.to_csv('../data/education2012.csv')
    return edu_latest


def parseEarning():
    earnings = pd.read_csv("../data/earnings.csv", encoding= "latin-1")
    features_earn = ['fips', 'total_med', 'total_agri_fish_mine', \
        'construction', 'manufacturing', 'wholesale_trade', 'retail_trade',\
                'utilities', 'information',\
                    'fin_ins_realest', \
                        'total_prof_sci_mgmt_admin', \
                            'total_edu_health_social', \
                                'total_arts_ent_acc_food',\
                                        'pub_admin']
    years = np.unique(earnings['year'])
    for year in years:
        earnings_year = earnings.loc[earnings['year'] == year]
        earnings_year = earnings_year[features_earn]
        earnings_year = earnings_year.convert_objects(convert_numeric=True)
        earnings_year.dropna(axis=0, inplace = True)
        filename = "../data/yearlyEarningData/earning_"+str(year)+".csv"
        earnings_year.to_csv(filename)
    earnings = earnings[features_earn]
    earnings = earnings.convert_objects(convert_numeric=True)
    earnings.dropna(axis=0, inplace = True)
    return earnings







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

