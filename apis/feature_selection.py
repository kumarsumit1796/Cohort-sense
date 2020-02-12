import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import lightgbm as lgm
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC

import warnings

warnings.filterwarnings("ignore")


class FeatureSelection:
    @staticmethod
    def selection(dataframe, functionName, dataarg):
        selectedFeatures = None

        if functionName == "correlation":
            print("Number of Features before Correlation " + str(len(dataframe.columns)))
            print("Enter the limit of Correlation Value")
            limit = float(dataarg)
            selectedFeatures = SelectionFunctions.correlation(dataframe, limit)

        elif functionName == "correlationPvalue":
            print("Number of Features before CorrelationPvalue " + str(len(dataframe.columns)))
            print("Enter the limit of Correlation Value")
            limit = float(dataarg)
            selectedFeatures = SelectionFunctions.correlation(dataframe, limit)
            selectedFeatures = SelectionFunctions.backwardElimination(dataframe, selectedFeatures)

        elif functionName == "LGBM":
            print("Number of Features before LGBM " + str(len(dataframe.columns)))
            print("Enter the Number of Features to be select")
            Number = int(dataarg)
            selectedFeatures = SelectionFunctions.lightGBM(dataframe, Number)

        elif functionName == "ChiSquare":
            print("Number of Features before ChiSquare " + str(len(dataframe.columns)))
            print("Enter the Number of Features to be select")
            Number = int(dataarg)
            selectedFeatures = SelectionFunctions.univariateSelection(dataframe, Number, "chi2")

        elif functionName == "fClassIf":
            print("Number of Features before fClassIf " + str(len(dataframe.columns)))
            print("Enter the Number of Features to be select")
            Number = int(dataarg)
            selectedFeatures = SelectionFunctions.univariateSelection(dataframe, Number, "f_classif")

        elif functionName == "recursiveFeatureElimination":
            print("Number of Features before RecursiveFeatureElimination " + str(len(dataframe.columns)))
            print("Enter the Number of Features to be select")
            Number = int(dataarg)
            selectedFeatures = SelectionFunctions.recursiveOrL1Based(dataframe, "recursiveFeatureElimination", Number)

        elif functionName == "l1basedFeatureSelection":
            print("Number of Features before l1basedFeatureSelection " + str(len(dataframe.columns)))
            selectedFeatures = SelectionFunctions.recursiveOrL1Based(dataframe, "l1basedFeatureSelection")

        elif functionName == "treeBasedFeatureSelection":
            print("Number of Features before treeBasedFeatureSelection " + str(len(dataframe.columns)))
            selectedFeatures = SelectionFunctions.recursiveOrL1Based(dataframe, "treeBasedFeatureSelection")

        return dataframe[selectedFeatures]


class SelectionFunctions:
    @staticmethod
    def correlation(dataframe, limit):
        corr = dataframe.corr()
        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i + 1, corr.shape[0]):
                if corr.iloc[i, j] >= limit or corr.iloc[i, j] <= -limit:
                    if columns[i]:
                        columns[i] = False
        selected_columns = list(dataframe.columns[columns])
        print("Number of Features after Correlation " + str(len(selected_columns)))
        return selected_columns

    @staticmethod
    def backwardElimination(dataframe, selectedFeatures):
        SL = 0.05
        corr_df = dataframe[selectedFeatures]
        x = corr_df.iloc[:, :-1].values
        Y = corr_df.iloc[:, -1].values

        corr_label = selectedFeatures[-1]
        columns = selectedFeatures[:-1]

        numVars = len(x[0])
        for i in range(0, numVars):
            regressor_OLS = sm.OLS(Y, x).fit()
            maxVar = max(regressor_OLS.pvalues).astype(float)
            if maxVar > SL:
                for j in range(0, numVars - i):
                    if regressor_OLS.pvalues[j].astype(float) == maxVar:
                        x = np.delete(x, j, 1)
                        columns = np.delete(columns, j)
        selected_columns = list(columns)
        selected_columns.append(corr_label)
        print("Number of Features after CorrelationPvalue " + str(len(selected_columns)))
        return selected_columns

    @staticmethod
    def lightGBM(dataframe, Number):
        target = dataframe.columns[-1]
        x = dataframe.drop([target], axis=1)
        y = dataframe.filter([target], axis=1)

        y[target] = y[target].astype(int)

        data = lgm.Dataset(x, label=y)
        params = {"max_depth": 15, "learning_rate": 0.1, "num_leaves": 900, "n_estimators": 100}
        lgm_model = lgm.train(params=params, train_set=data, categorical_feature='auto')

        this = list(lgm_model.feature_importance())
        effe_cols = []
        for i in this:
            if i != -1:
                effe_cols.append(lgm_model.feature_name()[this.index(i)])
                this[this.index(i)] = -1

        d = pd.DataFrame({'Score': this, 'Features': effe_cols})
        selected_cols = d.sort_values(by=['Score'], ascending=False).head(Number)['Features'].tolist()
        selected_cols.append('target')
        print("Number of Features after LightGBM " + str(len(selected_cols)))
        return selected_cols

    @staticmethod
    def univariateSelection(dataframe, Number, method):
        target = dataframe.columns[-1]
        X = dataframe.drop([target], axis=1)
        Y = dataframe.filter([target], axis=1)

        if method == "chi2":
            test = SelectKBest(score_func=chi2, k=Number)
        elif method == "f_classif":
            test = SelectKBest(score_func=f_classif, k=Number)
        selector = test.fit(X, Y)

        feature_names = list(X.columns.values)

        mask = selector.get_support()
        selected_features = []
        for bool, feature in zip(mask, feature_names):
            if bool:
                selected_features.append(feature)
        selected_features.append(target)
        print("Number of Features after " + method + " " + str(len(selected_features)))
        return selected_features

    @staticmethod
    def recursiveOrL1Based(dataframe, method, Number=None):
        target = dataframe.columns[-1]
        x = dataframe.drop([target], axis=1)
        y = dataframe.filter([target], axis=1)

        if method == "recursiveFeatureElimination":
            # Feature extraction
            model = LogisticRegression()
            rfe = RFE(model, Number)
            # Fit the function for ranking the features by score
            fit = rfe.fit(x, y)

        elif method == "l1basedFeatureSelection":
            lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(x, y)
            fit = SelectFromModel(lsvc, prefit=True)

        elif method == "treeBasedFeatureSelection":
            clf = ExtraTreesClassifier(n_estimators=50)
            clf = clf.fit(x, y)
            clf.feature_importances_
            fit = SelectFromModel(clf, prefit=True)

        # All features columns
        feature_names = list(x.columns.values)

        # Get the name of method selected features
        mask = fit.get_support()  # list of booleans
        selected_features = []  # The list of your K best features

        for bool, feature in zip(mask, feature_names):
            if bool:
                selected_features.append(feature)
        selected_features.append(target)
        print("Number of Features after recursive or L1 or tree based " + str(len(selected_features)))
        return selected_features
