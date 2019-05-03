import pandas as pd 
import os 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from KNN_county import *
import numpy as np
import matplotlib.pyplot as plt
# from weeklyDroughtMap import *

np.random.seed(seed=1234)

def make_county_adjacencyL():
    fname = "../data/county_adjacency.txt"
    latest = None
    adjacencyL = {}
    with open(fname,  encoding="latin=1") as f:
        for line in f:
            line = line.replace("\n", "")
            line = line.replace('"', "")
            lineL = line.split("\t")
            if lineL[0]:
                adjacencyL[int(lineL[1])] = {int(lineL[-1])}
                latest = int(lineL[1])
            else:
                adjacencyL[latest].add(int(lineL[-1]))
    return adjacencyL

def main():
    ad = make_county_adjacencyL()
    csv_files =  sorted(os.listdir("../data/Weekly Drought Level Graphs"))
    ############################################################
    # Cross Validation Analysis
    ############################################################

    numFolds = 8
    fnames = csv_files[len(csv_files)-numFolds-1:]

    sep_list = [i for i in range(0, 20, 1)]

    num_mistakesMat_baseline_test = np.empty((len(sep_list), numFolds))
    num_mistakesMat_baseline_train = np.empty((len(sep_list), numFolds))

    num_mistakesMat_knn_test = np.empty((len(sep_list), numFolds))
    num_mistakesMat_knn_train = np.empty((len(sep_list), numFolds))


    clf_base = Baseline()
    KNN_clfs = [KNN_county(ad, sep) for sep in sep_list]

    for i, fname  in enumerate(fnames[:numFolds]):
        fname="../data/Weekly Drought Level Graphs/"+fname
        data_cvFold = pd.read_csv(fname)
        data_X, data_y = data_cvFold['fips'].tolist(), data_cvFold['level_d'].tolist()

        X_train, X_test, y_train, y_test = train_test_split(data_X, data_y,  test_size=0.2)
        mse_baseline_test, mse_baseline_train = [], []
        mse_knn_test, mse_knn_train = [], []

        for j, sep in enumerate(sep_list):
            clf_base.fit(X_train, y_train)
            y_pred_test = clf_base.predict(X_test)
            y_pred_train = clf_base.predict(X_train)

            mse_baseline_test.append(clf_base.num_mistakes(y_pred_test, y_test))
            mse_baseline_train.append(clf_base.num_mistakes(y_pred_train, y_train))

            clf = KNN_clfs[j] 
            clf.fit(X_train, y_train)
            y_pred_test = clf.predict(X_test)
            y_pred_train = clf.predict(X_train)

            mse_knn_test.append(clf.num_mistakes(y_pred_test, y_test))
            mse_knn_train.append(clf.num_mistakes(y_pred_train, y_train))

        num_mistakesMat_knn_train[:,i] = mse_knn_train
        num_mistakesMat_knn_test[:,i] = mse_knn_test 

        num_mistakesMat_baseline_train[:,i] = mse_baseline_train
        num_mistakesMat_baseline_test[:,i] = mse_baseline_test
        print("fold", i, "done") 

    mse_mean_score_knn_train = np.mean(num_mistakesMat_knn_train, axis = 1)
    mse_mean_score_knn_test = np.mean(num_mistakesMat_knn_test, axis = 1)
    mse_mean_score_baseline_train = np.mean(num_mistakesMat_baseline_train, axis = 1)
    mse_mean_score_baseline_test = np.mean(num_mistakesMat_baseline_test, axis = 1)

    lab_names = ['KNN Train', "KNN Test", "Baseline Train", "Baseline Test"]
    all_scores = [mse_mean_score_knn_train, mse_mean_score_knn_test, mse_mean_score_baseline_train, mse_mean_score_baseline_test]
    for i, scores in enumerate(all_scores):
        plt.plot(sep_list, scores, label = lab_names[i])
    plt.ylim([0, max(mse_mean_score_baseline_test) + 0.1*max(mse_mean_score_baseline_test)])
    plt.legend(loc = 'best')
    plt.title("Cross Validation Analysis")
    plt.xlabel("Degrees of Separation")
    plt.ylabel("Error Rate")
    plt.savefig("drought_classifier_plots/CV_knn_num_mistakes.png")
    plt.close()

    best_clf = KNN_clfs[np.argmin(mse_mean_score_knn_test)]
    print("Best Degree of Separation:", sep_list[np.argmin(mse_mean_score_knn_test)])

    ############################################################
    # Learning Curve Analysis 
    ############################################################

    test_set_sizes = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    mse_baseline_test, mse_baseline_train = [], []
    mse_knn_test, mse_knn_train = [], []

    clf_base = Baseline()

    fname = fnames[1]
    fname="../data/Weekly Drought Level Graphs/"+fname
    data_cvFold = pd.read_csv(fname)
    data_X, data_y = data_cvFold['fips'].tolist(), data_cvFold['level_d'].tolist()

    for test_set_size in test_set_sizes:
        X_train, X_test, y_train, y_test = train_test_split(data_X, data_y,  test_size=test_set_size)

        best_clf.fit(X_train, y_train)
        y_pred_test = best_clf.predict(X_test)
        y_pred_train = best_clf.predict(X_train)

        mse_knn_test.append(best_clf.num_mistakes(y_pred_test, y_test))
        mse_knn_train.append(best_clf.num_mistakes(y_pred_train, y_train))

        print("test size", test_set_size, "done")

    lab_names = ['KNN Train', "KNN Test"]
    all_scores = [mse_knn_train, mse_knn_test]
    training_set_sizes = 1-np.array(test_set_sizes)
    for i, scores in enumerate(all_scores):
        plt.plot(training_set_sizes, scores, label = lab_names[i])

    plt.ylim([0, max(mse_knn_test)+0.1* max(mse_knn_test)])
    plt.legend(loc = 'best')
    plt.title("Learning Curve Analysis")
    plt.xlabel("Trainning Set Size")
    plt.ylabel("Error Rate")
    plt.savefig("drought_classifier_plots/LC_knn_num_mistakes.png")



    
    fname_test = fnames[-1]

    date = fname_test.split("_")[-1]
    date = date.split(".")[0]
    print(date)

    fname_test_full ="../data/Weekly Drought Level Graphs/"+fname_test
    data_test  = pd.read_csv(fname_test_full)
    data_X_real, data_y_real = data_test['fips'].tolist(), data_test['level_d'].tolist()
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(data_X_real, data_y_real,  test_size=0.2)

    # trainningSet = pd.concat([X_train, y_train], axis = 1)

    best_clf.fit(X_train_real, y_train_real)
    y_pred_real = best_clf.predict(X_test_real)
    # print(y_pred)
    print(best_clf.num_mistakes(y_pred_real, y_test_real))


    fname_split = fname_test.split(".")[-2] 
    # just training 
    trainningSet = pd.DataFrame({'fips': X_train_real, 'level_d': y_train_real})
    fname_training = ".."+ fname_split +"_training.csv"
    trainningSet.to_csv(fname_training)

    # just gt
    ground_truth = pd.DataFrame({ 'fips' :  X_test_real, 'level_d': y_test_real})
    fname_ground_truth = ".."+ fname_split +"_ground.csv" 
    ground_truth.to_csv(fname_ground_truth)

    # just predicted
    predict_df = pd.DataFrame({ 'fips' : X_test_real, 'level_d': y_pred_real})
    fname_test_pred = ".."+ fname_split +"_pred.csv"
    predict_df.to_csv(fname_test_pred)

    # full gt and prediected
    full_pred_df = pd.DataFrame({ 'fips' : X_train_real + X_test_real, 'level_d': y_train_real + y_test_real, 'level_d_predict': y_train_real + y_pred_real})
    fname_test_fullpred = ".."+fname_split+"_fullpred.csv"
    full_pred_df.to_csv(fname_test_fullpred)



    title = 'Drought level for week of '+ str(date) 
    # makeOneDroughtPNG(fname_training, 'drought_classifier_plots/training.png', 'level_d',title + ' | Traing', save = True)
    # makeOneDroughtPNG(fname_ground_truth, 'drought_classifier_plots/ground_truth.png', 'level_d', title + ' Ground Truth', save = True)


    # makeOneDroughtPNG(fname_test_pred, 'drought_classifier_plots/predicted.png', 'level_d', title + ' | Predicted', save = True)

    # makeOneDroughtPNG(fname_test_fullpred, 'drought_classifier_plots/full_ground_truth.png', 'level_d', title + ' | Full Ground Truth', save = True)
    # makeOneDroughtPNG(fname_test_fullpred, 'drought_classifier_plots/full_predicted.png', 'level_d_predict', title +  ' | Full Predicted', save = True)

if __name__ == "__main__":
    main()
    # pass
