from datetime import datetime
import time

import numpy as np
import pandas as pd
import pickle


test_data_address = 'test_data.csv'
model_address = 'knn.model'
temporal_features = ['C1_School closing', 'C2_Workplace closing', 'C3_Cancel public events',
                     'C4_Restrictions on gatherings', 'C5_Close public transport',
                     'C6_Stay at home requirements', 'C7_Restrictions on internal movement',
                     'C8_International travel controls', 'H1_Public information campaigns',
                     'H2_Testing policy', 'H3_Contact tracing', 'H6_Facial Coverings']
fixed_feature = 'weekday'
target = 'ConfirmedCases'


def read_data(address):
    data = pd.read_csv(address)
    X = pd.read_csv(address)
    X[fixed_feature] = X['Date'].map(lambda x: 0 if datetime.strptime(str(x), '%Y%M%d').weekday() < 5 else 1)
    X.drop(columns=[column for column in X.columns.values
                    if column not in temporal_features + [fixed_feature] + [target]],
           inplace=True)
    return data, X


def feature_preprocess(data):
    for column in temporal_features + [fixed_feature]:
        data[column].replace(' ', np.NaN, inplace=True)
        data[column] = data[column].astype(float)
        data[column].fillna(data[column].mode()[0], inplace=True)
    return data


def load_model():
    knn_model = pickle.load(open(model_address, 'rb'))
    return knn_model


def main():
    test_data, X = read_data(test_data_address)
    X = feature_preprocess(X)
    X = X[temporal_features + [fixed_feature]]
    X = X.to_numpy()
    knn_model = load_model()
    y_pred = np.round(knn_model.predict(X)).astype(np.int)
    np.savetxt("prediction.csv", y_pred, delimiter=",")
    test_data['prediction'] = pd.Series(data=y_pred)
    test_data.to_csv('test_data_prediction.csv')


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('Elapsed Time =', end_time - start_time, 's')
