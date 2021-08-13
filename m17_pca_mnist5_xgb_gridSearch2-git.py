# Grid, RandomSearch with mnist

params = [
    {"n_estimators":[100, 200, 300], 
    "learning_rate":[0.001, 0.01],
    "max_depth":[4, 5, 6], 
    "colsample_bytree":[0.6, 0.9, 1], 
    "colsample_bylevel":[0.6, 0.7, 0.9],
    "n_jobs":[-1]}
]

# Practice : n_comp upper than 0.999 : 486
# make model -> Tensorflow DNN, compare with banila

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.datasets import mnist

from sklearn.decomposition import PCA
import warnings 
warnings.filterwarnings(action='ignore')

(x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

x = np.append(x_train, x_test, axis=0) # (70000, 28, 28)
y = np.append(y_train, y_test, axis=0) # (70000,)

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

pca = PCA(n_components=486)
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)

cumsum = np.cumsum(pca_EVR)
# print(cumsum)

# print(np.argmax(cumsum >= 0.95)+1) 

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.14, shuffle=True, random_state=77)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = QuantileTransformer()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, GlobalAveragePooling2D, Dropout
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, cross_val_score, train_test_split

model = RandomizedSearchCV(XGBClassifier(), params, verbose=1)

import time

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time

# 4. predict eval -> no need to

print('totla time : ', end_time)
print('Best estimator : ', model.best_estimator_)
print('Best score  :', model.best_score_)