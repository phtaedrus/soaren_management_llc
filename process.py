# python3

import csv
import pandas as pd
import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Use 3 decimal places in output display
pd.set_option("display.precision", 3)

# Don't wrap repr(DataFrame) across additional lines
pd.set_option("display.expand_frame_repr", True)

# Set max rows displayed in output to 25
pd.set_option("display.max_rows", 30)

# Change the style of matplotlib
plt.style.use('dark_background')

FILE = "data/Soaren_Management_Lead_Bid_Test_Data.csv"
SAMPLE_SIZE = 30


class CustomDataFrame:
    def __init__(self, file):
        self.my_df = self.get_df_from_csv(file)
        self.headers = self.add_headers(file)
        self.custom_df = self.my_df.reindex(columns=self.headers)
        self.math_data = None
        self.sample_size = np.random

    def get_df_from_csv(self, file: csv) -> pd.DataFrame():
        with open(FILE) as f:
            df = pd.read_csv(file)

        self.my_df = df

        return df

    def add_headers(self, file: csv) -> []:
        with open(FILE) as f:
            lines = f.readlines()

        headers = lines[0].strip().split(sep=',')
        headers[3], headers[4] = headers[4], headers[3]

        self.headers = headers

        return headers

    def do_the_maths(self):
        df = self.custom_df
        math_df = df.drop(columns='id')
        math_df['AcceptedBid'] = math_df['AcceptedBid'].replace([0, 0], np.nan)

        math_df = math_df.dropna()
        print(math_df.sample(30))



        def custom_regression(__df: pd.DataFrame):
            _df = __df.copy()


            print(_df.nunique())

            """split into different dataframe buckets for overlays and individual regression"""
            a = __df.loc[__df['BidPrice'] == 3.0]
            b = __df.loc[__df['BidPrice'] == 35.0]
            c = __df.loc[__df['BidPrice'] == 50.0]
            d = __df.loc[__df['BidPrice'] == 75.0]

            _df = pd.concat([a, b, c, d], axis=0).replace('NaN', 0).drop(columns='AcceptedBid')

            print(_df)

            #plt.scatter(x=a.ExpectedConversion, y=a.ExpectedRevenue, color='magenta', alpha=0.5)
            #plt.scatter(x=b.ExpectedConversion, y=b.ExpectedRevenue, color='red', alpha=0.5)
            #plt.scatter(x=c.ExpectedConversion, y=c.ExpectedRevenue, color='blue', alpha=0.5)
            #plt.scatter(x=d.ExpectedConversion, y=d.ExpectedRevenue, color='green', alpha=0.5)

            #plt.legend(labels=['3.0', '35.0', '50.0', '75.0'])
            _df['ExpectedNetRevenue'] = ((1 * _df.ExpectedConversion) * _df.ExpectedRevenue) - _df.BidPrice
            _df['PotentialMargin(%)'] = (_df.BidPrice / _df.ExpectedRevenue) * 100.0
            print(_df)

            X_0 = _df['BidPrice'].values.reshape(-1, 1)
            X = _df['PotentialMargin(%)'].values.reshape(-1, 1)
            y = _df['ExpectedNetRevenue'].values.reshape(-1, 1)

            #adjust between X_0 and X currently for training results. Multi-V_Regression will account for
            #more coefficients in future source code.
            X_train, X_test, y_train, y_test = train_test_split(X_0, y, test_size=0.2, random_state=0)

            m_lr = LinearRegression()
            m_lr. fit(X_train, y_train)

            # To retrieve the intercept:
            print(m_lr.intercept_)
            # For retrieving the slope:
            print(m_lr.coef_)

            y_hat = m_lr.predict(X_test)
            #y_hat2 = m_lr.predict(y_test)

            df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_hat.flatten()})
            print(df.sample(25))
            df1 = df.sample(25)

            df1.plot(kind='bar', figsize=(16, 10))

            plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
            plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

            plt.show()

            #plt.scatter(X_test, y_test, color='gray')
            #plt.plot(X_test, y_hat, color='red', linewidth=2)
            #plt.ylabel('ActualExpectedRevenue', color='green')
            #plt.xlabel('PredictedRevenue', color="yellow")
            #plt.show()

            #print(df1.sample(30))



        self.math_data = math_df

        return [math_df, custom_regression(math_df)]


def main():
    a = CustomDataFrame(FILE)
    # print(a.custom_df)
    a.do_the_maths()
    # a.plot_the_data()


if __name__ == '__main__':
    main()
