import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_absolute_error, ConfusionMatrixDisplay
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.optimizers import Adagrad
df = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bikes.csv')

def preprocess(X):
    for col in X.select_dtypes(include='object'):
        try:
            pd.to_numeric(X[col])
            X[col] = pd.to_numeric(X[col])
        except ValueError:
            pass
    # Drop highly unique columns
    for col in X.columns:
        if X[col].dtype == 'object':
            unique_vals = X[col].nunique()
            unique_ratio = unique_vals / len(X[col])
            if unique_ratio >= 0.8:
                X = X.drop(col, axis=1)

    X['dteday'] = pd.to_datetime(X['dteday'])
    X['day'] = X['dteday'].dt.day
    X['month'] = X['dteday'].dt.month
    X['year'] = X['dteday'].dt.year
    X['dayofweek'] = X['dteday'].dt.dayofweek

    X = X.drop('dteday', axis=1)

    # Scale data
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X

X = df.drop(['casual', 'registered'], axis=1)\
    .pipe(preprocess)
y = df.assign(y = df['casual'] + df['registered'])['y']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

# Build and compile the neural network
model = Sequential()

# Add input layer
model.add(Dense(10, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))


early_stopping = EarlyStopping(monitor='val_loss', patience=5)
#  0.0015
adam = Adam()
model.compile(loss='mse', optimizer=adam, metrics=['mean_absolute_error'])


batch_size_var = 64

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0)

history = model.fit(X_train, 
    y_train, 
    epochs=10000, 
    batch_size=batch_size_var, 
    validation_data=(X_val, y_val), 
    callbacks=[early_stopping])


# Make predictions on testing data
y_pred = model.predict(X_test)
# Find regression test metrics
r2 = r2_score(y_test, y_pred)
mean_absolute_error = mean_absolute_error(y_test, (y_pred))
metric = "r2 is "+str(r2)+" and mean absolute error is "+str(mean_absolute_error)
print(metric)

model.save('r2_8706_mae_44.935')

# df_hold = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bikes_december.csv')

# hold_X, hold_y = preprocess(df_hold)