from datetime import datetime
import time

import numpy as np
import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
# from sklearn.svm import SVR
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.models import Sequential
import pickle

train_data_address = 'train_data.csv'
test_data_address = 'test_data.csv'
temporal_features = ['C1_School closing', 'C2_Workplace closing', 'C3_Cancel public events',
                     'C4_Restrictions on gatherings', 'C5_Close public transport',
                     'C6_Stay at home requirements', 'C7_Restrictions on internal movement',
                     'C8_International travel controls', 'H1_Public information campaigns',
                     'H2_Testing policy', 'H3_Contact tracing', 'H6_Facial Coverings']
fixed_feature = 'weekday'
target = 'ConfirmedCases'


def read_data(address):
    data = pd.read_csv(address)
    data[fixed_feature] = data['Date'].map(lambda x: 0 if datetime.strptime(str(x), '%Y%M%d').weekday() < 5 else 1)
    data.drop(columns=[column for column in data.columns.values
                       if column not in temporal_features + [fixed_feature] + [target]],
              inplace=True)
    return data


def feature_preprocess(data):
    for column in temporal_features + [fixed_feature]:
        data[column].replace(' ', np.NaN, inplace=True)
        data[column] = data[column].astype(float)
        data[column].fillna(data[column].mode()[0], inplace=True)
    return data


def target_preprocess(data):
    data[target].replace(' ', np.NaN, inplace=True)
    data[target].fillna(0, inplace=True)
    data[target] = data[target].astype(float)
    data = data[data[target] > 0]
    return data


def feature_target_split(data):
    X_train = data[temporal_features + [fixed_feature]].to_numpy()
    y_train = data[target].to_numpy()
    return X_train, y_train


def train_train_validation_split(X_train, y_train):
    X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(X_train,
                                                                              y_train,
                                                                              test_size=0.3,
                                                                              random_state=13)
    return X_train_train, X_train_val, y_train_train, y_train_val


def knn(X_train_train, y_train_train):
    knn_model = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
    knn_model.fit(X_train_train, y_train_train)
    return knn_model


# def svm(X_train_train, y_train_train):
#     svm_model = SVR()
#     svm_model.fit(X_train_train, y_train_train)
#     return svm_model
#
#
# def rf(X_train_train, y_train_train):
#     rf_model = RandomForestRegressor(n_estimators=100, verbose=0)
#     rf_model.fit(X_train_train, y_train_train)
#     return rf_model
#
#
# def glm(X_train_train, y_train_train):
#     glm_model = LinearRegression()
#     glm_model.fit(X_train_train, y_train_train)
#     return glm_model
#
#
# def nn(X_train_train, y_train_train):
#     y_train_train = y_train_train.reshape(y_train_train.shape[0], 1)
#     nn_model = Sequential()
#     nn_model.add(Dense(8, input_dim=X_train_train.shape[1], activation='relu'))
#     nn_model.add(Dense(16, activation='relu'))
#     nn_model.add(Dense(8, activation='relu'))
#     nn_model.add(Dense(1, activation='linear'))
#     nn_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#     nn_model.fit(X_train_train, y_train_train, epochs=100, batch_size=8, verbose=1, validation_split=0.2)
#     return nn_model


def prediction(model, X):
    y_pred = model.predict(X)
    y_pred = y_pred.reshape(-1)
    return y_pred


def mape(y, y_pred):
    sumOfAbsoluteError = sum(abs(y - y_pred))
    percentageOfAbsoluteError = (sumOfAbsoluteError / sum(y)) * 100
    return percentageOfAbsoluteError


def main():
    train_data = read_data(train_data_address)
    test_data = read_data(test_data_address)
    train_data = feature_preprocess(train_data)
    test_data = feature_preprocess(test_data)
    train_data = target_preprocess(train_data)
    test_data = target_preprocess(test_data)
    X_train, y_train = feature_target_split(train_data)
    X_test, y_test = feature_target_split(test_data)
    X_train_train, X_train_val, y_train_train, y_train_val = train_train_validation_split(X_train, y_train)

    for model, method in model_method.items():
        model_dict[model]['model'] = method(X_train_train, y_train_train)
        model_dict[model]['y_pred_train_val'] = np.round(prediction(model_dict[model]['model'], X_train_val))
        model_dict[model]['train_val_mape'] = mape(y_train_val, model_dict[model]['y_pred_train_val'])
        model_dict[model]['y_pred_test'] = np.round(prediction(model_dict[model]['model'], X_test))
        model_dict[model]['test_mape'] = mape(y_test, model_dict[model]['y_pred_test'])
        print(model, model_dict[model]['train_val_mape'], model_dict[model]['test_mape'])
        with open(model + '.model', 'wb') as model_pickle:
            pickle.dump(model_dict[model]['model'], model_pickle)
    model_pickle = pickle.load(open('knn.model', 'rb'))
    y_hat = np.round(model_pickle.predict(X_test))


if __name__ == '__main__':
    start_time = time.time()
    # model_method = {'knn': knn, 'svm': svm, 'rf': rf, 'glm': glm, 'nn': nn}
    model_method = {'knn': knn}
    model_dict = {model_name: {'model': None, 'y_pred_train_val': None, 'y_pred_test': None,
                               'train_val_mape': None, 'test_mape': None}
                  for model_name in model_method.keys()}
    main()
    end_time = time.time()
    print('Elapsed Time =', end_time - start_time, 's')
