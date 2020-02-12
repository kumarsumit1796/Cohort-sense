import pandas as pd
import seaborn as se
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score, recall_score, precision_score, \
    accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle
import time
import warnings
warnings.filterwarnings("ignore")


class ModelSelection:

    @staticmethod
    def split_train_test(df, tts):
        labels = df['target'].values
        features = df.drop(['target'], axis=1).values
        print("Enter the test set split size else press <enter> to move on. It takes default as 0.3")
        test_ss = tts
        if len(test_ss) != 0:
            train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                                        test_size=float(test_ss),
                                                                                        random_state=1)
            train_test = [train_features, test_features, train_labels, test_labels]
        else:
            train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3,
                                                                                        random_state=1)
            train_test = [train_features, test_features, train_labels, test_labels]
        return train_test

    @staticmethod
    def traditional_models():
        model_list = []
        logr = LogisticRegression()
        rfc = RandomForestClassifier()
        abc = AdaBoostClassifier()
        model_list.append(('logistic_regression', logr))
        model_list.append(('random_forest', rfc))
        model_list.append(('adaboost', abc))
        return model_list

    @staticmethod
    def predict_model(model, train_test, mname, flag):
        fit_model = model.fit(train_test[0], train_test[2])
        model_predict = fit_model.predict(train_test[1])
        if flag==1:
            mbtime = str(int(time.time()))
            picmodelname = mname+mbtime+'.pkl'
            pathh = r"c:\Users\sathickibrahims\Desktop\Sathick\CohortAnalyzer\CohortAnalyzer"
            model_path = os.path.join(pathh, 'pickeldir', picmodelname)
            model_pickle = open(model_path, 'wb')
            pickle.dump(fit_model, model_pickle)
            model_pickle.close()
            print('prediction done')
            return model_predict, picmodelname
        else:
            print('prediction done')
            return model_predict

    @staticmethod
    def model_metrics(test_labels, model_predict, possitive_proba):
        model_accuracy = round(accuracy_score(test_labels, model_predict), 2)
        model_recall = round(recall_score(test_labels, model_predict), 2)
        model_precision = round(precision_score(test_labels, model_predict), 2)
        roc_auc_score_model = round(roc_auc_score(test_labels, possitive_proba), 2)
        print('accuracy found')
        return model_accuracy, model_recall, model_precision, roc_auc_score_model

    # Function to perform probability prediction with desired percentage
    @staticmethod
    def probability_prediction(model, train_test, percent=0.9, default=False, positive_proba=False):
        test_features = train_test[1]
        model_predict = ModelSelection.predict_model(model, train_test, "", 0)
        probability_1 = model.predict_proba(test_features)[:, 1]
        probability_0 = model.predict_proba(test_features)[:, 0]
        if default is False:
            p1_gt = [p for p in probability_1 if p >= percent]
            p0_gt = [p for p in probability_0 if p >= percent]
            if positive_proba is True:
                return p1_gt, p0_gt, probability_1
            return p1_gt, p0_gt
        else:
            p1_gt90 = [p90 for p90 in probability_1 if p90 >= .9]
            p0_gt90 = [p90 for p90 in probability_0 if p90 >= .9]
            p1_gt95 = [p95 for p95 in probability_1 if p95 >= .95]
            p0_gt95 = [p95 for p95 in probability_0 if p95 >= .95]
            p1_gt99 = [p99 for p99 in probability_1 if p99 >= .99]
            p0_gt99 = [p99 for p99 in probability_0 if p99 >= .99]
            print("Number of patients having 90% probability of predicting 1: ", len(p1_gt90))
            print("Number of patients having 90% probability of predicting 0: ", len(p0_gt90))
            print("Number of patients having 95% probability of predicting 1: ", len(p1_gt95))
            print("Number of patients having 95% probability of predicting 0: ", len(p0_gt95))
            print("Number of patients having 99% probability of predicting 1: ", len(p1_gt99))
            print("Number of patients having 99% probability of predicting 0: ", len(p0_gt99))
            p1_per = ['90%', '95%', '99%']
            p1_val = [len(p1_gt90), len(p1_gt95), len(p1_gt99)]
            p1_set = {'Probability': p1_per, 'count': p1_val}
            p1_df = pd.DataFrame(p1_set)
            se.barplot(x=p1_df['Probability'], y=p1_df['count'])
            # plt.title('Possitive Probability')

    # function to compare different ML models:
    @staticmethod
    def compare_models(train_test):
        scn = []
        sc = []
        from matplotlib.pyplot import figure
        modelname = []
        modelaccuracy = []
        model_roc_auc_score = []
        model_precision_list = []
        model_recall_list = []
        model_cm_list = []
        model_pic_name = []
        figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        train_features = train_test[0]
        test_features = train_test[1]
        train_labels = train_test[2]
        test_labels = train_test[3]
        model_list = ModelSelection.traditional_models()

        for model_name, select_model in model_list:
            model_predict, picmodelname = ModelSelection.predict_model(select_model, train_test, model_name, 1)
            # Predict Probability
            print('Predict Probability')
            probability_1, probability_0, possitive_proba = ModelSelection.probability_prediction(select_model,
                                                                                                  train_test, 0.9,
                                                                                                  positive_proba=True)
            print("Results for: " + model_name)
            print("P1: ", len(probability_1))
            print("P0: ", len(probability_0))
            model_accuracy, model_recall, model_precision, roc_auc_score_model = ModelSelection.model_metrics(
                test_labels, model_predict, possitive_proba)
            model_msg = "%s %f %f" % (model_name, model_accuracy, roc_auc_score_model)
            model_confu = confusion_matrix(test_labels, model_predict)
            model_pic_name.append(picmodelname)
            modelname.append(model_name)
            modelaccuracy.append(model_accuracy)
            model_roc_auc_score.append(roc_auc_score_model)
            model_precision_list.append(model_precision)
            model_recall_list.append(model_recall)
            scn.append(model_name)
            sc.append(roc_auc_score_model)
            test_labels_binary = label_binarize(test_labels, classes=[0, 1])
            fpr, tpr, threshold = roc_curve(test_labels_binary, possitive_proba)
            roc_auc = auc(fpr, tpr)
            # plt.plot(fpr, tpr, label=(model_name, round(roc_auc, 2)))
            # plt.legend(loc='upper left')
            print(model_name + ' Acc: ' + str(model_accuracy) + ' AUC: ' + str(roc_auc_score_model) + ' Recall: ' + str(
                model_recall) + ' Precision: ' + str(model_precision) + ' FPR: ' + str(
                round(((model_confu[0][1]) / (model_confu[0][1] + model_confu[0][0])), 2)) + ' TPR: ' + str(
                round(((model_confu[1][1]) / (model_confu[1][1] + model_confu[1][0])), 2)))
            print('TN , FP \nFN, TP')
            print(model_confu)
            model_cm_list.append(model_confu.tolist())
        # plt.title("ROC_AUC CURVE", weight='semibold', size=20)
        return scn, sc, [modelaccuracy, model_roc_auc_score, model_recall_list, model_precision_list], model_cm_list, model_pic_name

    @staticmethod
    def select_model(modelname):
        if modelname == 'logistic_regression':
            model = LogisticRegression()
            return model
        elif modelname == 'random_forest':
            model = RandomForestClassifier()
            return model
        elif modelname == 'adaboost':
            model = AdaBoostClassifier()
            return model
        else:
            print('model not found')

    @staticmethod
    def run_model(model, param_dict, train_test, verbose=True, n_jobs=-1):
        clf = GridSearchCV(model, param_dict, verbose=verbose, n_jobs=n_jobs, error_score=0.0).fit(train_test[0],
                                                                                                   train_test[2])
        print("Training score : ", clf.score(train_test[0], train_test[2]))
        print("Testing score : ", clf.score(train_test[1], train_test[3]))
        print("Best params : ", clf.best_params_)
        mbp = clf.best_params_
        return clf, mbp

    @staticmethod
    def run_model_wt_par(model_obj, train_test, model_name, oacc, oauc, orecall, oprecision):
        model_predict,picmodelname = ModelSelection.predict_model(model_obj, train_test, model_name+"AT",1)
        print(oacc)
        print('Predict Probability')
        probability_1, probability_0, possitive_proba = ModelSelection.probability_prediction(model_obj, train_test,
                                                                                              0.9, positive_proba=True)
        print("P1: ", len(probability_1))
        print("P0: ", len(probability_0))
        model_accuracy, model_recall, model_precision, roc_auc_score_model = ModelSelection.model_metrics(train_test[3],
                                                                                                          model_predict,
                                                                                                          possitive_proba)
        model_confu = confusion_matrix(train_test[3], model_predict)
        print("Before Model Tunning, For " + model_name + " the metrics are Acc " + str(oacc) + " AUC " + str(
            oauc) + " Recall " + str(orecall) + " Precision " + str(oprecision))
        print(
            "After Model Tunning, For " + model_name + ' the metrics are Acc: ' + str(model_accuracy) + ' AUC: ' + str(
                roc_auc_score_model) + ' Recall: ' + str(model_recall) + ' Precision: ' + str(model_precision))
        print('TN , FP \nFN, TP')
        print(model_confu)
        return model_name, model_accuracy, roc_auc_score_model, model_recall, model_precision, model_confu,picmodelname

    @staticmethod
    def run_tuning(modelname, train_test, all_met, all_cm):
        if modelname == 'logistic_regression':
            model = ModelSelection.select_model(modelname='logistic_regression')
            params = {
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'C': [x for x in range(1, 5)]  # np.linespace(0.1,10,100)
            }
            LRparams, tun_par = ModelSelection.run_model(model, params, train_test)
            print("Fine Tunned logistic_regression Model")
            lr_acc = all_met[0][0]
            lr_auc = all_met[1][0]
            lr_recall = all_met[2][0]
            lr_preci = all_met[3][0]
            lr_tp = LogisticRegression(C=tun_par['C'], penalty=tun_par['penalty'], solver=tun_par['solver'])
            mn, ma, rasm, mr, mp, mc, picmodelname = ModelSelection.run_model_wt_par(lr_tp, train_test, "logistic_regression", lr_acc,
                                                                       lr_auc, lr_recall, lr_preci)
            return mn, ma, rasm, mr, mp, mc, picmodelname

        elif modelname == 'random_forest':
            model = ModelSelection.select_model(modelname='random_forest')
            params = {
                'bootstrap': [True, False],
                'max_depth': [x for x in range(5, 40, 8)],
                'criterion': ['gini', 'entropy'],
                'max_features': ['auto', 'sqrt', 'log2'],
                'min_samples_leaf': [1, 2, 3],
                'min_samples_split': [2, 4, 6],
                'n_estimators': [10, 100, 500, 1000, 1500, 2000, 3000]
            }
            RFCparams, tun_par = ModelSelection.run_model(model, params, train_test)
            print("Fine Tunned random_forest Model")
            rfc_acc = all_met[0][1]
            rfc_auc = all_met[1][1]
            rfc_recall = all_met[2][1]
            rfc_preci = all_met[3][1]
            # rfc_otpr = round(((all_cm[1][1][1])/(all_cm[1][1][1]+all_cm[1][1][0])),2)
            # rfc_ofpr = round(((all_cm[1][0][1])/(all_cm[1][0][1]+all_cm[1][0][0])),2)
            rfc_tp = RandomForestClassifier(bootstrap=tun_par['bootstrap'], max_depth=tun_par['max_depth'],
                                            criterion=tun_par['criterion'], max_features=tun_par['max_features'],
                                            min_samples_leaf=tun_par['min_samples_leaf'],
                                            min_samples_split=tun_par['min_samples_split'],
                                            n_estimators=tun_par['n_estimators'])
            mn, ma, rasm, mr, mp, mc, picmodelname = ModelSelection.run_model_wt_par(rfc_tp, train_test, "random_forest", rfc_acc,
                                                                       rfc_auc, rfc_recall, rfc_preci)
            return mn, ma, rasm, mr, mp, mc, picmodelname

        elif modelname == 'adaboost':
            model = ModelSelection.select_model(modelname='adaboost')
            params = {
                'n_estimators': [10, 100, 500, 1000, 1500, 2000],
                'learning_rate': [0.01, 0.1, 1],
                'base_estimator': [GaussianNB(), LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier(),
                                   SVC()]
            }
            ABCparams, tun_par = ModelSelection.run_model(model, params, train_test)
            print("Fine Tunned adaboost Model")
            abc_acc = all_met[0][2]
            abc_auc = all_met[1][2]
            abc_recall = all_met[2][2]
            abc_preci = all_met[3][2]
            # abc_otpr = round(((all_cm[2][1][1])/(all_cm[2][1][1]+all_cm[2][1][0])),2)
            # abc_ofpr = round(((all_cm[2][0][1])/(all_cm[2][0][1]+all_cm[2][0][0])),2)
            abc_tp = AdaBoostClassifier(n_estimators=tun_par['n_estimators'], learning_rate=tun_par['learning_rate'],
                                        base_estimator=tun_par['base_estimator'])
            mn, ma, rasm, mr, mp, mc, picmodelname = ModelSelection.run_model_wt_par(abc_tp, train_test, "adaboost", abc_acc, abc_auc,
                                                                       abc_recall, abc_preci)
            return mn, ma, rasm, mr, mp, mc, picmodelname
