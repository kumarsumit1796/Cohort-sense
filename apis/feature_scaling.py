from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


class FeatureScaling:
    @staticmethod
    def Scaling(dataframe, functionname):
        if functionname == "standardScaler":
            scaler = StandardScaler()
        elif functionname == "minMaxScaler":
            scaler = MinMaxScaler()
        elif functionname == "maxAbsScaler":
            scaler = MaxAbsScaler()
        elif functionname == "normalizer":
            scaler = Normalizer()
        elif functionname == "robustScaler":
            scaler = RobustScaler()
        features_columns = list(dataframe.columns[:-1])
        dataframe[features_columns] = scaler.fit_transform(dataframe[features_columns])
        return dataframe
