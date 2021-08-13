# 실습
# mnist 데이터를 pca를 통해 cnn으로 구성
# (28,28) -> 784 -> 차원축소 (625) -> (25, 25) -> CNN 모델 구성

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.datasets import mnist

from sklearn.decomposition import PCA

(x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

x = np.append(x_train, x_test, axis=0) # (70000, 28, 28)
y = np.append(y_train, y_test, axis=0) # (70000,)

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

pca = PCA(n_components=400)
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_

cumsum = np.cumsum(pca_EVR)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.14, shuffle=True, random_state=77)


from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer, OneHotEncoder
scaler = QuantileTransformer()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x = x.reshape(x.shape[0], 20, 20, 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, GlobalAveragePooling2D, Dropout

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5),                          
                        padding='same', activation='relu' ,input_shape=(20, 20, 1))) 
model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))                   
model.add(MaxPool2D())                                         
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))                   
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))    
model.add(Flatten())                                              
model.add(Dense(256, activation='relu'))
model.add(Dense(124, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor= 'loss', patience=50, mode='min', verbose=1)

import time

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=100, batch_size=512, callbacks=[es], validation_split=0.1, verbose=2)

end_time = time.time() - start_time

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('걸린 시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])


'''
pca 미적용
걸린 시간 :  79.20276927947998
loss :  0.1189129501581192
accuracy :  0.9814304709434509

pca 적용
걸린 시간 :  69.4980092048645
loss :  0.12350702285766602
accuracy :  0.9740842580795288
'''