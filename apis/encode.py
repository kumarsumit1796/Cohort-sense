import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")


class DataEncode:
    @staticmethod
    def label_encode(data, col_name):
        encode = LabelEncoder()
        data[col_name] = encode.fit_transform(data[col_name].astype('str'))
        return data

    @staticmethod
    def one_hot_encode(data, col_name):
        dummies = pd.get_dummies(data[col_name], prefix=col_name)
        data = pd.concat([data, dummies], axis=1)
        del data[col_name]
        return data
