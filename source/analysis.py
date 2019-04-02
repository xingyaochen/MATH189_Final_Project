from clustering import *
from regression import *
from load import * 

def main():
    ################## DATA MUNGING TIME ################################

    features_earn = ['total_med', 'total_agri_fish_mine', \
            'construction', 'manufacturing', 'wholesale_trade', 'retail_trade',\
                    'utilities', 'information',\
                        'fin_ins_realest', \
                            'total_prof_sci_mgmt_admin', \
                                'total_edu_health_social', \
                                    'total_arts_ent_acc_food',\
                                            'pub_admin']

    withdraw_categories = ['pub_sup_10', 'dom_sup_4', 'ind_7', \
                'irrigation_1', 'livestock_3', \
                'aqua_7', 'mining_7', 'thermoelectric_27',\
                    'gro_wat_1', 'surf_wat_1',
                    ]
    merge_water(withdraw_categories)
    final_merge = pd.read_csv('../data/drought-usage'+str(year)+".csv")

    ################## CLUSTERING TIME ################################

    model = KMeans(n_clusters=5, n_init = 200, random_state = 1234)

    final_merge_cluster, training_earn = cluster_by_earning(final_merge, model, verbose=True)
    
    features  = list(training_earn)
    training_earn['cluster'] = model.labels_
    for i in range(len(features)):
        for j in range(len(features)):
            if i != j:
                plot_cluster(training_earn, features[i], features[j])

    ################## 1D DUMB LINEAR REGRSSION TIME #####################
    clusters = np.unique(model.labels_)
    for c in clusters:
        data = extract_cluster(final_merge_cluster, c, withdraw_categories)
        for withdraw_category in withdraw_categories:
            withdraw_means = get_mean_effect(data, withdraw_category, c, plot = True, log = True)
            slope_sum = get_slope(withdraw_means, withdraw_category)
            if slope_sum < -0.3:
                print(slope_sum)
                print("cluster", c)
                print('withdraw category', withdraw_category)
                print("\n")

    ################## RIDGE REGRESSION TIME ################################
    keep_fips = final_merge['fips']
    training_earn, labels_earn = read_earning_reg(keep_fips = keep_fips)
    training_earn['fips'] = labels_earn
    clusters = np.unique(model.labels_)
    reg_data = final_merge_cluster[['fips']+withdraw_categories + ['cluster']]
    reg_data = reg_data.merge(training_earn, left_on = 'fips', right_on='fips')
    reg_data.dropna(axis=0, inplace = True)

    for c in clusters:
        reg_data_cluster = reg_data.loc[reg_data['cluster'] == c]
        X = reg_data_cluster[withdraw_categories]
        reg_model = linear_model.RidgeCV(alphas =[0.001, 0.1, 1.0, 10.0, 100.0], cv= 8, normalize = True)
        y = reg_data_cluster['total_med']
        reg_model.fit(X, y)
        print('best alpha:', reg_model.alpha_)
        print('coefs:',reg_model.coef_)

        highest_ind = np.argsort(reg_model.coef_)[::-1]

        f, ax = plt.subplots(1, 2, sharey=True, figsize = (20, 10))

        for i, h_ind in enumerate(highest_ind[:2]):
            feature = list(X)[h_ind]
            x = X[feature]
            this_ax = ax[i]
            this_ax.scatter(np.log(x), y, s = 10)
            abline(reg_model.coef_[h_ind], reg_model.intercept_, this_ax)
            this_ax.set_xlabel(feature+ " (logged)", fontsize=20)
            if i == 0:
                this_ax.set_ylabel("Total Median Earning", fontsize=20)

        f.suptitle("Linear Regression of Earning ~ Water Withdraw | Cluster "+ str(c), fontsize=28)
        plt.savefig('plots4/'+ "linreg_c"+ str(c)+".png")
        plt.close()

if __name__ == "__main__":
    main()



