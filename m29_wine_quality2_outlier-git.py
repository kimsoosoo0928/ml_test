# 실습
# 아웃라이어 확인


import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_wine
from xgboost import XGBClassifier

#DATA
datasets = pd.read_csv('./_data/winequality-white.csv',
                        index_col=None, header=0, sep=';')

print(datasets.head())
print(datasets.shape) # (4898, 12)
print(datasets.describe())

datasets = datasets.values
print(type(datasets)) # <class 'numpy.ndarray'>

x = datasets[:, :11]
y = datasets[:, 11]

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

#################################재사용##################################
def outliers(data_out):
    quantile_1, q2, quantile_3 = np.percentile(data_out, [25, 50, 75])
    print("1사분위 : ", quantile_1)
    print("q2 : ", q2)
    print("3사분위 : ", quantile_3)
    iqr = quantile_3 - quantile_3
    lower_bound = quantile_1 - (iqr * 1.5)
    upper_bound = quantile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out>lower_bound))
outliers_loc = outliers(x_train)
print("이상치의 위치 : ", outliers_loc)
# 아웃라이어의 갯수를 count하는 기능 추가할것!
########################################################################

import matplotlib.pyplot as plt

plt.boxplot(x_train)
plt.show()

# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# #MODEL
# model = XGBClassifier(n_jobs=-1)

# model.fit(x_train, y_train)

# score = model.score(x_test, y_test)

# print("acc : ", score) # acc :  0.6816326530612244