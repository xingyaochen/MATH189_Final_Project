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




withdraw_categories = ['pub_sup_10', 'dom_sup_4', 'ind_7', \
                'irrigation_1', 'livestock_3', \
                'aqua_7', 'mining_7', 'thermoelectric_27',\
                    'gro_wat_1', 'surf_wat_1',
                    ]


features_earn = ['total_med', 'total_agri_fish_mine', \
        'construction', 'manufacturing', 'wholesale_trade', 'retail_trade',\
                'utilities', 'information',\
                    'fin_ins_realest', \
                        'total_prof_sci_mgmt_admin', \
                            'total_edu_health_social', \
                                'total_arts_ent_acc_food',\
                                        'pub_admin']


final_merge = pd.read_csv('../data/drought-usage'+str(year)+".csv")

keep_fips = final_merge['fips']
model = KMeans(n_clusters=5, n_init = 200, random_state = 1234)
final_merge_cluster, _ = cluster_by_earning(final_merge, model, verbose=True)

training_earn, labels_earn = read_earning_reg(keep_fips = keep_fips)

training_earn['fips'] = labels_earn


clusters = np.unique(model.labels_)

reg_data = final_merge_cluster[['fips']+withdraw_categories + ['cluster']]

reg_data = reg_data.merge(training_earn, left_on = 'fips', right_on='fips')

reg_data.dropna(axis=0, inplace = True)



# if log:
#         data_nonzero = data.loc[ data[withdraw_category] > 0]
#         X = data_nonzero['level_d']
#         y = np.log(data_nonzero[withdraw_category]) 
#         data_nonzero[withdraw_category] = y



# for c in clusters:
#     reg_data_cluster = reg_data.loc[reg_data['cluster'] == c]
#     X = reg_data_cluster[withdraw_categories]
#     reg_model = linear_model.RidgeCV(alphas =[0.001, 0.1, 1.0, 10.0, 100.0, 100.0], cv= 8, normalize = True)
#     y = reg_data_cluster['total_med']
#     reg_model.fit(X, y)
#     print('best alpha:', reg_model.alpha_)
#     print('coefs:',reg_model.coef_)
#     print('\n')



def abline(slope, intercept, axes):
    """Plot a line from slope and intercept"""
    # axes = ax.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    axes.plot(x_vals, y_vals, 'r-', color="red")



for c in clusters:
    reg_data_cluster = reg_data.loc[reg_data['cluster'] == c]
    X = reg_data_cluster[withdraw_categories]
    reg_model = linear_model.RidgeCV(alphas =[0.001, 0.1, 1.0, 10.0, 100.0, 100.0], cv= 8, normalize = True)
    y = reg_data_cluster['total_med']
    reg_model.fit(X, y)
    print('best alpha:', reg_model.alpha_)
    print('coefs:',reg_model.coef_)

    highest_ind = np.argsort(reg_model.coef_)[::-1]

    f, ax = plt.subplots(1, 3, sharey=True, figsize = (20, 5))


    for i, h_ind in enumerate(highest_ind[:3]):
        feature = list(X)[h_ind]
        x = X[feature]
        this_ax = ax[i]
        # this_ax.set_aspect(0.000001)
        this_ax.scatter(np.log(x), y, s = 10)
        abline(reg_model.coef_[h_ind], reg_model.intercept_, this_ax)
        this_ax.set_xlabel(feature+ " (logged)")
        if i == 0:
            this_ax.set_ylabel("Total Median Earning")

    f.suptitle("Linear Regression of Earning ~ Water Withdraw | Cluster "+ str(c))
    plt.savefig('plot4/'+ "linreg_c"+ str(c)+".png")
    plt.close()