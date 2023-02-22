import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_absolute_error, ConfusionMatrixDisplay
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bikes.csv')

df = df\
    .assign(y = lambda x: x['registered'] + x['casual'] - x['registered'])\
    .drop(['registered', 'casual'], axis=1)

# Assign target variable
y = pd.DataFrame(df['y'])

# Drop target variable from features
X = df.drop('y', axis=1)

# Turn all numeric object columns into numeric columns
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



# Make all columns numerical
X = X.pipe(pd.get_dummies)

# Scale data
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

# Build and compile the neural network
model = Sequential()

# Add input layer
model.add(Dense(10, activation='relu', input_dim=X_train.shape[1]))

# Add hidden layers
for i in range(2):
    model.add(Dense(10, activation='relu'))

# Add output layer
model.add(Dense(y_train.shape[1], activation='linear'))

# Define loss variable
loss_var = 'mse'

#Compile model  
model.compile(loss=loss_var, optimizer='adam', metrics=['mean_absolute_error'])

# Train model
batch_size_var = 64
history = model.fit(X, y, epochs=100, batch_size=batch_size_var)

# Evaluate the neural network on training data
model.evaluate(X, y)

# Make predictions on testing data
y_pred = model.predict(X_test)

# Find regression test metrics
r2 = r2_score(y_test, y_pred)
mean_absolute_error = mean_absolute_error(y_test, y_pred)
metric = "r2 is "+str(r2)+" and mean absolute error is "+str(mean_absolute_error)

print(metric)