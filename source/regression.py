from clustering import * 
from sklearn import linear_model


def read_earning_reg(keep_fips, scale = False):
    earnings = pd.read_csv("../data/earnings.csv", encoding= "latin-1")
    earnings2010 = earnings.loc[earnings['year'] == year]
    earnings2010 = earnings2010.loc[earnings2010['fips'].isin(keep_fips)]
    features_earn = ['fips', 'total_med', 'total_agri_fish_mine', \
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


def abline(slope, intercept, axes):
    """Plot a line from slope and intercept"""
    # axes = ax.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    axes.plot(x_vals, y_vals, 'r-', color="red")


