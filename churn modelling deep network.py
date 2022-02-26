import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


pd.set_option('display.max_columns', 500)

df = pd.read_csv("")


X = df.iloc[:, 3:-1].values

y = df.iloc[:, -1].values

print(X[:,1])
print(len(X[0]))

le = LabelEncoder()
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')

X[:, 2] = le.fit_transform(X[:,2])
X = np.array(ct.fit_transform(X))

print(X[0])
print(len(X[0]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train[0]


model = tf.keras.Sequential([
    tf.keras.Input(shape=(12,)),
    tf.keras.layers.Dense(6, activation = 'relu'),
    tf.keras.layers.Dense(6, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

model.compile(loss = 'binary_crossentropy', optimizer ='adam', metrics = ['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
train_model = model.fit(X_train, y_train, epochs = 100, validation_data =(X_test, y_test), callbacks =[callback])
