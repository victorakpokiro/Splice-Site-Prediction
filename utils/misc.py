import numpy as np

class Normalization(object):

    @staticmethod
    def min_std_norm(df):
        for col in df.columns:
            c = df[col]
            df[col] = (c-c.mean())/c.std()
        return df

class Metrics(object):

    @staticmethod
    def mse(y_true, y_predicted):
        return np.mean((y_true - y_predicted)**2)

