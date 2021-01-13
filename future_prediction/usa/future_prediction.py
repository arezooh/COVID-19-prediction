from futuremakeHistoricalData import futuremakeHistoricalData
from makeHistoricalData import makeHistoricalData
from models import GBM, GLM, KNN, NN, MM_GLM, GBM_grid_search, NN_grid_search
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import time
import os
import random
import tensorflow as tf
from numpy.random import seed
import requests
import matplotlib.pyplot as plt
from pexecute.process import ProcessLoom
import subprocess
from zipfile import ZipFile
from sys import argv

address = './'
if int(argv[1]) == 1 :
    data_address = '../csvFiles/'
else :
    data_address = '../csvFiles/weatherless/'

seed(1)
tf.random.set_seed(1)

numberOfSelectedCounties = -1
target_name = 'death'
spatial_mode = 'country'
country_name = 'US'
future_mode = False
pivot = 'country'
test_size = 28
max_history = 14
future_features = []
selected_futures = []

first_run = 1
# 'd' shows daily mode, 'w' shows weekly mode, 's' shows weekly mode with scenario testing,
# 'm' shows daily mode with LSTM MM_GLM, 'b' shows daily mode with best method obtained from search
# and numbers shows 'r' (number of days in future)
run_list=[]#'w28','d14','m14','b14',,'s28'

daily_r = [14]
weekly_r = []

best_methods={'d14':'MM_GLM'}#,'m14':'LSTM_MIXED','b14':'MM_GLM','w28':'KNN','s28':'KNN'
county_errors = {error:None for error in ['MAPE','MASE','MAE']}
country_errors = {error:None for error in ['MAPE','MASE','MAE']}

best_h={'d14':3}#,'m14':1,'b14':3,'w28':2,'s28':1
best_c={'d14':1}#,'m14':3,'b14':1,'w28':1,'s28':9


best_scaler = 1

all_covariates = ['death',
       'social-distancing-total-grade', 'daily-state-test',
       'social-distancing-encounters-grade', 'confirmed',
       'social-distancing-travel-distance-grade', 'precipitation',
       'temperature', 'weekend']#


none_mixed_methods = ['GBM', 'GLM', 'KNN', 'NN','LSTM']
mixed_methods = ['MM_GLM', 'MM_NN','LSTM_MIXED']
models_to_log = ['NN', 'GLM', 'GBM', 'KNN']#
best_loss = {'GBM': 'least_absolute_deviation', 'MM_NN': 'MeanAbsoluteError', 'NN': 'MeanAbsoluteError'}#MeanAbsoluteError


temporal_data = pd.read_csv(data_address + 'temporal-data.csv')


temporal_data['date'] = temporal_data['date'].apply(lambda x : datetime.datetime.strptime(x, '%y/%m/%d'))

zero_removing = 1
# if pivot == 'country':
#     zero_removing = 0

######################################################### split data to train, val, test
def splitData(numberOfCounties, main_data, target, spatial_mode, mode, r, temporal_mode, gap_flag):
    numberOfCounties = len(main_data['county_fips'].unique())
    main_data = main_data.sort_values(by=['date of day t', 'county_fips'])
    target = target.sort_values(by=['date of day t', 'county_fips'])
    # we set the base number of days to the minimum number of days existed between the counties
    # and then compute the validation size for the non-default state.
    baseNumberOfDays = (main_data.groupby(['county_fips']).size()).min()
    test_size = r
    
    if temporal_mode == 'weekly':
        val_size = round(0.3 * (baseNumberOfDays - test_size))#49 
        test_size = 1
    else :
        val_size = 21
        
    if gap_flag == 1:
        gap = r
    else :
        gap = 1

    if mode == 'val':
        
        X_test = main_data.tail(test_size * numberOfCounties).copy()
        X_train_val = main_data.iloc[:-((test_size + gap-1) * numberOfCounties)].tail(val_size * numberOfCounties).copy()
        X_train_train = main_data.iloc[:-((val_size + test_size + gap-1) * numberOfCounties)].copy()

        y_test = target.tail(test_size * numberOfCounties).copy()
        y_train_val = target.iloc[:-((test_size + gap-1) * numberOfCounties)].tail(val_size * numberOfCounties).copy()
        y_train_train = target.iloc[:-((val_size + test_size + gap-1) * numberOfCounties)].copy()

        return X_train_train, X_train_val, X_test, y_train_train, y_train_val, y_test

    if mode == 'test':
        
        X_test = main_data.tail(test_size * numberOfCounties).copy()
        X_train = main_data.iloc[:-((test_size + gap-1) * numberOfCounties)].copy()

        y_test = target.tail(test_size * numberOfCounties).copy()
        y_train = target.iloc[:-((test_size + gap-1) * numberOfCounties)]

        return X_train, X_test, y_train, y_test
    
    
########################################################### clean data
def clean_data(data, numberOfSelectedCounties, spatial_mode):
    global numberOfDays
    data = data.sort_values(by=['county_fips', 'date of day t'])
    # select the number of counties we want to use
    # numberOfSelectedCounties = numberOfCounties
    if numberOfSelectedCounties == -1:
        numberOfSelectedCounties = len(data['county_fips'].unique())

    using_data = data[(data['county_fips'] <= data['county_fips'].unique()[numberOfSelectedCounties - 1])]
    using_data = using_data.reset_index(drop=True)
    if (spatial_mode == 'county') or (spatial_mode == 'country'):
        if pivot == 'county' :
            main_data = using_data.drop(['county_name', 'state_fips', 'state_name'],
                                        axis=1)  # , 'date of day t'
        elif pivot == 'state':
            main_data = using_data.drop(['county_name'],
                                        axis=1)  # , 'date of day t'
        elif pivot == 'country':
            main_data = using_data

    elif (spatial_mode == 'state'):
        main_data = using_data.drop(['county_name', 'state_name'],
                                    axis=1)
    numberOfDays = len(using_data['date of day t'].unique())

    return main_data


########################################################### preprocess
def preprocess(main_data, spatial_mode, validationFlag, r, temporal_mode, gap_flag):
    
    if spatial_mode == 'state':
        target = pd.DataFrame(main_data[['date of day t', 'county_fips', 'state_fips', 'Target']])
    else:
        target = pd.DataFrame(main_data[['date of day t', 'county_fips', 'Target']])

    main_data = main_data.drop(['Target'], axis=1)

    # produce train, validation and test data
    if validationFlag:  # validationFlag is 1 if we want to have a validation set and 0 otherwise

        X_train_train, X_train_val, X_test, y_train_train, y_train_val, y_test = splitData(numberOfSelectedCounties,
                                                                                           main_data, target, spatial_mode,
                                                                                           'val', r, temporal_mode, gap_flag)
        return X_train_train, X_train_val, X_test, y_train_train, y_train_val, y_test

    else:

        X_train, X_test, y_train, y_test = splitData(numberOfSelectedCounties, main_data, target, spatial_mode,
                                                     'test', r, temporal_mode, gap_flag)
        return X_train, X_test, y_train, y_test
    
########################################################### logarithmize covariates
def logarithm_covariates(data):
    # make temporal and some fixed covariates logarithmic
#     if temporal_mode == 'daily':
    negative_features = ['temperature', 'Retail', 'Grocery', 'Parks', 'Transit', 'Workplace', 'Residential']
    # for weekly average mode we dont logarithm the target so its bether not to logarithm its history
#     elif temporal_mode == 'weekly':
#       negative_features = ['temperature']#,target_name
    
    for covar in data.columns:
        if (' t' in covar) and (covar.split(' ')[0] not in negative_features) and (
                covar not in ['county_fips', 'date of day t']):
            data[covar] = np.log((data[covar] + 1).astype(float))

    fix_log_list = ['total_population', 'population_density', 'area', 'median_household_income',
                    'houses_density', 'airport_distance', 'deaths_per_100000']
    for covar in fix_log_list:
        if covar in data.columns:
            data[covar] = np.log((data[covar] + 1).astype(float))
    return(data)
    
########################################################### run algorithms in parallel except mixed models
def run_algorithms(X_train, X_val, y_train, y_val, best_loss, algorithm , mode):
    
    # from models import GBM, GLM, KNN, NN
    
    if mode =='test': y_val[pd.isnull(y_val['Target'])]['Target']=1 # it doesnt have values for test mode and we set these values to 1 to preventing errors
        
    y_prediction={method:None for method in none_mixed_methods+mixed_methods}
    y_prediction_train={method:None for method in none_mixed_methods+mixed_methods}
    Xtrain={method:None for method in none_mixed_methods+mixed_methods}
    Xval={method:None for method in none_mixed_methods+mixed_methods}
    
    X_train = X_train.drop(['county_fips', 'date of day t'], axis=1)
    X_val = X_val.drop(['county_fips', 'date of day t'], axis=1)
    y_train = np.array(y_train['Target']).reshape(-1)
    y_val = np.array(y_val['Target']).reshape(-1)
    
    for method in none_mixed_methods:
        Xtrain[method] = X_train
        Xval[method] = X_val
        if method in models_to_log:
            Xtrain[method] = logarithm_covariates(Xtrain[method])
            Xval[method] = logarithm_covariates(Xval[method])
        
    if algorithm == 'GBM' or algorithm in mixed_methods:
        y_prediction['GBM'], y_prediction_train['GBM'] = GBM(Xtrain['GBM'], Xval['GBM'], y_train, best_loss['GBM'])
        
    if algorithm == 'GLM' or algorithm in mixed_methods:
        y_prediction['GLM'], y_prediction_train['GLM'] = GLM(Xtrain['GLM'], Xval['GLM'], y_train)
        
    if algorithm == 'KNN' or algorithm in mixed_methods:
        y_prediction['KNN'], y_prediction_train['KNN'] = KNN(Xtrain['KNN'], Xval['KNN'], y_train)
        
    if algorithm == 'NN' or algorithm in mixed_methods:
        y_prediction['NN'], y_prediction_train['NN'] = NN(Xtrain['NN'], Xval['NN'], y_train, y_val, best_loss['NN'])
        
    if algorithm == 'LSTM' or algorithm == 'LSTM_MIXED':
        y_prediction['LSTM'], y_prediction_train['LSTM'] = LSTMM(Xtrain['LSTM'], Xval['LSTM'], y_train, y_val)

#     print('y_prediction[NN]',y_prediction['NN'])
#     print('y_prediction[LSTM]',y_prediction['LSTM'])
    
    if algorithm in mixed_methods:
        
        y_predictions_test, y_predictions_train = [], []
        # Construct the outputs for the testing dataset of the 'MM' methods
        y_predictions_test.extend([y_prediction['GBM'], y_prediction['GLM'], y_prediction['KNN'],
                                   y_prediction['NN']])
        y_prediction_test_np = np.array(y_predictions_test).reshape(len(y_predictions_test), -1)
        X_test_mixedModel = pd.DataFrame(y_prediction_test_np.transpose())
        # Construct the outputs for the training dataset of the 'MM' methods
        y_predictions_train.extend(
            [y_prediction_train['GBM'], y_prediction_train['GLM'], y_prediction_train['KNN'],
             y_prediction_train['NN']])
        y_prediction_train_np = np.array(y_predictions_train).reshape(len(y_predictions_train), -1)
        X_train_mixedModel = pd.DataFrame(y_prediction_train_np.transpose())
        
        if algorithm == 'MM_GLM':
            y_prediction['MM_GLM'], y_prediction_train['MM_GLM'] = GLM(X_train_mixedModel, X_test_mixedModel, y_train)
        elif algorithm == 'MM_NN':
            y_prediction['MM_NN'], y_prediction_train['MM_NN'] = NN(X_train_mixedModel, X_test_mixedModel, y_train, y_val, best_loss['NN'])
    
    
    if algorithm == 'LSTM_MIXED':
        
        y_predictions_test, y_predictions_train = [], []
        # Construct the outputs for the testing dataset of the 'MM' methods
        y_predictions_test.extend([y_prediction['GBM'], y_prediction['GLM'], y_prediction['KNN'],
                                   y_prediction['NN'],y_prediction['LSTM']])
        y_prediction_test_np = np.array(y_predictions_test).reshape(len(y_predictions_test), -1)
        X_test_mixedModel = pd.DataFrame(y_prediction_test_np.transpose())
        # Construct the outputs for the training dataset of the 'MM' methods
        y_predictions_train.extend(
            [y_prediction_train['GBM'], y_prediction_train['GLM'], y_prediction_train['KNN'],
             y_prediction_train['NN'], y_prediction_train['LSTM']])
        y_prediction_train_np = np.array(y_predictions_train).reshape(len(y_predictions_train), -1)
        X_train_mixedModel = pd.DataFrame(y_prediction_train_np.transpose())
#         print(X_train_mixedModel)
        y_prediction['LSTM_MIXED'], y_prediction_train['LSTM_MIXED'] = GLM(X_train_mixedModel, X_test_mixedModel, y_train)
        
    return(y_prediction[algorithm], y_prediction_train[algorithm])

########################################################### get errors for each model in each h and c
def get_errors(y_prediction, y_prediction_train, y_test_date, y_train_date, regular_data, numberOfSelectedCounties, r, temporal_mode):
    
    # y_test_date and y_train_date are a dataframes with columns ['date of day t', 'county_fips', 'Target']
    # set negative predictions to zero
    y_prediction[y_prediction < 0] = 0
    y_test = np.array(y_test_date['Target']).reshape(-1)
    
    county_errors = {error: None for error in
                      ['MAE', 'MAPE','MASE']}
    # country_errors show error for prediction of target variable for whole country
    country_errors = {error: None for error in
                      ['MAE', 'MAPE','MASE']}

    if numberOfSelectedCounties == -1:
        numberOfSelectedCounties = len(y_test_date['county_fips'].unique())
        
    ##################################### MASE denominator
    X_train_train, X_train_val, X_test, mase_y_train_train_date, mase_y_train_val_date, mase_y_test_date = preprocess(regular_data,
                                                                                                       spatial_mode, 1, r, temporal_mode)

    train_train = (mase_y_train_train_date.reset_index(drop=True)).sort_values(by=['date of day t', 'county_fips'])
    train_val = (mase_y_train_val_date.reset_index(drop=True)).sort_values(by=['date of day t', 'county_fips'])

    df = train_train.append(train_val)

    train_val = df.tail(len(train_val))
    train_val = train_val.rename(columns={'Target': 'val-Target', 'date of day t': 'val-date'})
#     print('mase1',train_val)

    train_train = df.iloc[:-(r*numberOfSelectedCounties),:].tail(len(train_val))
    train_train = train_train.rename(
        columns={'Target': 'train-Target', 'date of day t': 'train-date'})
#     print('mase2',train_train)
    

    df_for_train_val_MASE_denominator = pd.concat(
        [train_train.reset_index(drop=True), train_val.reset_index(drop=True)], axis=1)
    df_for_train_val_MASE_denominator['absolute-error'] = abs(df_for_train_val_MASE_denominator['val-Target'] -
                                                              df_for_train_val_MASE_denominator['train-Target'])
    train_val_MASE_denominator = df_for_train_val_MASE_denominator['absolute-error'].mean()

    # we need to have mase denominator based on target values for whole country (sum of target for all counties)
    # this will be used for calculation of country error
    df_for_train_val_MASE_denominator_country = df_for_train_val_MASE_denominator.groupby(['val-date']).sum()
    df_for_train_val_MASE_denominator_country['absolute-error'] = abs(
        df_for_train_val_MASE_denominator_country['val-Target'] -
        df_for_train_val_MASE_denominator_country['train-Target'])

    train_val_MASE_denominator_country = df_for_train_val_MASE_denominator_country['absolute-error'].mean()
    #####################################


    # if target mode is logarithmic we need to return the target variable to its original state
    if target_mode == 'logarithmic':
        print('logarithmic')
        y_test = np.array(np.round(np.exp(y_test) - 1)).reshape(-1)
        y_test_date['Target'] = list(np.round(np.exp(y_test_date['Target']) - 1))
        y_prediction = np.array(np.exp(y_prediction) - 1).reshape(-1)

    # make predictions rounded to their closest number
    y_prediction = np.array(y_prediction)
    if target_mode != 'weeklyaverage':
        y_prediction = np.round(y_prediction)
    # for calculating the country error we must sum up all the county's target values to get country target value
    y_test_date['prediction'] = y_prediction
    y_test_date.to_csv('errors.csv')
    y_test_date_country = y_test_date.groupby(['date of day t']).sum()
    y_test_country = np.array(y_test_date_country['Target']).reshape(-1)
    y_prediction_country = np.array(y_test_date_country['prediction']).reshape(-1)
    
    ############################################################## calculate whole country error
    min_error = 1e10
    best_scaler = 1
    for i in range(10):
        print(i+1)
        error = mean_absolute_error(y_test_country, np.array(y_prediction_country)*(i+1))
        if error < min_error :
            min_error = error
            best_scaler = i+1
    print('best_scaler: ',best_scaler)
    # best_scaler = 1
    y_prediction = np.array(y_prediction)*best_scaler
    country_errors['MAE'] = mean_absolute_error(y_test_country, y_prediction_country)
    sumOfAbsoluteError = sum(abs(y_test_country - y_prediction_country))
    country_errors['MAPE'] = (sumOfAbsoluteError / sum(y_test_country)) * 100
    y_test_temp_country = y_test_country.copy()
    y_test_temp_country[y_test_country == 0] = 1
    y_prediction_temp_country = y_prediction_country.copy()
    y_prediction_temp_country[y_test_country == 0] += 1
    
    MASE_numerator = sumOfAbsoluteError / len(y_test_country)
    country_errors['MASE'] = MASE_numerator / train_val_MASE_denominator_country
    

    ############################################################## calculate county error
    y_prediction = np.array(y_prediction)*best_scaler
    county_errors['MAE'] = mean_absolute_error(y_test, y_prediction)
    print("Mean Absolute Error of ", county_errors['MAE'])
    sumOfAbsoluteError = sum(abs(y_test - y_prediction))
    county_errors['MAPE'] = np.mean((abs(y_test - y_prediction)/y_test)*100)#(sumOfAbsoluteError / sum(y_test)) * 100
    # we change zero targets into 1 and add 1 to their predictions
    y_test_temp = y_test.copy()
    y_test_temp[y_test == 0] = 1
    y_prediction_temp = y_prediction.copy()
    y_prediction_temp[y_test == 0] += 1
    # meanPercentageOfAbsoluteError = sum((abs(y_prediction_temp - y_test_temp) / y_test_temp) * 100) / len(y_test)
    print("Percentage of Absolute Error ", county_errors['MAPE'])
    MASE_numerator = sumOfAbsoluteError / len(y_test)
    county_errors['MASE'] = MASE_numerator / train_val_MASE_denominator
    print("MASE Error of ", county_errors['MASE'])

    print("-----------------------------------------------------------------------------------------")

    # # save outputs in 'out.txt'
    # sys.stdout = orig_stdout
    # f.close()
    
    return county_errors, country_errors, best_scaler

###################################### make dates nominal
def date_nominalize(data,temporal_mode,column_name):
    if column_name == 'Date':
        if temporal_mode == "daily" :
            data['Date'] = data['Date'].apply(lambda x: datetime.datetime.strftime(x,"%d %b %Y"))
        elif temporal_mode == "weekly" :
            date_list = np.array(data.sort_values(by=['Date'])['Date'].unique()).astype('datetime64[s]').tolist()
            weeks = data.sort_values(by=['Date'])['Date'].unique()
            for i,week in enumerate(weeks) :
                week_end_day = date_list[i]
                week_first_day = week_end_day - datetime.timedelta(days=6)
                week_dates = week_first_day.strftime("%d %b")+' -- '+week_end_day.strftime("%d %b %Y")
                data['Date'] = data['Date'].replace(week,week_dates)
    if column_name == 'Prediction Date':
        data['Prediction Date'] = data['Prediction Date'].apply(lambda x: datetime.datetime.strptime(x,'%y/%m/%d').strftime("%d %b %Y"))
    return(data)

###################################################### make data weekly
def make_weekly(dailydata):
    dailydata.sort_values(by=['date','county_fips'],inplace=True)
    numberofcounties=len(dailydata['county_fips'].unique())
    numberofweeks=len(dailydata['date'].unique())//7

    weeklydata=pd.DataFrame(columns=dailydata.columns)

    for i in range(numberofweeks):
        temp_df=dailydata.head(numberofcounties*7) # weekly average of last week for all counties
        date=temp_df.tail(1)['date'].iloc[0]
        dailydata=dailydata.iloc[(numberofcounties*7):,:]
        temp_df=temp_df.drop(['date'],axis=1)
        temp_df=temp_df.groupby(['county_fips']).mean().reset_index()
        temp_df['date']=date # last day of week 
        weeklydata=weeklydata.append(temp_df)
    weeklydata.sort_values(by=['county_fips','date'],inplace=True)
    weeklydata=weeklydata.reset_index(drop=True)
    return(weeklydata)


############################################## getting updated real values from data source
def get_csv(web_addres,file_address):
    url=web_addres
    print(url)
    req = requests.get(url)
    url_content = req.content
    csv_file = open(file_address, 'wb')
    csv_file.write(url_content)
    csv_file.close

############################################## getting updated real values from data source
def correct_negative_numbers(data):
    data2=data.copy()
    for i in range(5,data.shape[1]):
        data2.iloc[:,i]=data.iloc[:,i]-data.iloc[:,i-1]
    data=data2.copy()
    reverse_dates=data.columns[4:][::-1]
    while data.iloc[:,4:].min().sum()<0:
        for i in range(len(reverse_dates)):
            date = reverse_dates[i]
            negative_index=data[data[date]<0].index
            data2.loc[negative_index,date] = 0
        for i in range(len(reverse_dates)):
            date = reverse_dates[i]
            past_date = reverse_dates[i+1]
            negative_index=data[data[date]<0].index
            data2.loc[negative_index,past_date] = data2.loc[negative_index,past_date]+data.loc[negative_index,date]
            if i==len(reverse_dates)-2:
                break
        data=data2.copy()
    return(data)

def get_updated_covid_data(address):
    get_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    ,address + 'international-covid-death-data.csv')
    get_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    ,address + 'international-covid-confirmed-data.csv')

    

def generate_updated_temporal_data(address):
    death = pd.read_csv(address+'international-covid-death-data.csv')
    confirmed = pd.read_csv(address+'international-covid-confirmed-data.csv')
    death=death[death['Country/Region']=='US'].T
    death=death.iloc[4:,:]
    death.iloc[1:]=(death.iloc[1:].values-death.iloc[:-1].values)
    death = death.reset_index()
    death.columns = ['date','death']
    death['death']= death['death'].astype(int)
    confirmed=confirmed[confirmed['Country/Region']=='US'].T
    confirmed=confirmed.iloc[4:,:]
    confirmed.iloc[1:]=(confirmed.iloc[1:].values-confirmed.iloc[:-1].values)
    confirmed = confirmed.reset_index()
    confirmed.columns = ['date','confirmed']
    confirmed['confirmed']= confirmed['confirmed'].astype(int)
    confirmed_death = pd.merge(death,confirmed)
    confirmed_death['county_fips']=1
    confirmed_death['date'] = confirmed_death['date'].apply(lambda x:datetime.datetime.strptime(x,'%m/%d/%y'))
    confirmed_death=confirmed_death.sort_values(by=['date'])
    return(confirmed_death)

def add_real_values(data,address,temporal_mode):
    data['prediction datetime'] = data['Prediction Date'].apply(lambda x : datetime.datetime.strptime(x,"%Y-%m-%d"))
    data['datetime'] = data['Date'].apply(lambda x : datetime.datetime.strptime(x[-11:],"%d %b %Y"))
    temporal_data = generate_updated_temporal_data(address)

    for date in data['Prediction Date'].unique():
      if pd.isna(data[data['Prediction Date']==date].tail(1)['Real'].values):
        previous_prediction_date = datetime.datetime.strptime(date,"%Y-%m-%d")
        
        temp = temporal_data[['county_fips','date',target_name]]

        real_values_df = pd.DataFrame(columns=['date',target_name])
        
        if temporal_mode == 'daily':
            first_day = previous_prediction_date
            last_day = first_day + datetime.timedelta(days=14)
            daily_part = temp[(temp['date']>first_day) & (temp['date']<=last_day)]
            if len(daily_part)>0 : last_day = max(daily_part['date'])
            real_values_df = real_values_df.append(daily_part)

        if temporal_mode == 'weekly':
            first_day = previous_prediction_date
            print(first_day)
            last_day = first_day + datetime.timedelta(days=70)
            print(last_day)
            weekly_part = temp[(temp['date']>first_day) & (temp['date']<=last_day)]
#             print(weekly_part)
            if len(weekly_part)>0 : last_day = max(weekly_part['date'])
            weekly_part = make_weekly(weekly_part)
            real_values_df = real_values_df.append(weekly_part)
#             print('real_values_df')
#             print(real_values_df)

        data = pd.merge(data, real_values_df, how='left', left_on=['datetime','county_fips'],
                        right_on=['date','county_fips'])
#         print(data)
        new_ind = ~pd.isnull(data[target_name])|~pd.isnull(data['Absolute Error'])
#         print(new_ind)
        # add real values
        data.loc[~pd.isnull(data[target_name]),'Real'] = data.loc[~pd.isnull(data[target_name]),target_name]
        # calculate daily errors
        data.loc[new_ind,'Absolute Error']=abs(data.loc[new_ind,'Real']-data.loc[new_ind,'Prediction'])

        # calculate average errors
        for i in data[new_ind].index.values:

            # mean absolute error
            past_data = data.loc[:i,:]
            future_days_index = past_data[~pd.isna(past_data['Absolute Error'])].index
            present_sum_absolute_error = past_data['Absolute Error'].sum()
            present_number = len(future_days_index)
            sum_of_absolute_error = present_sum_absolute_error# + past_sum_absolute_error
            sum_of_numbers = present_number# + past_number
            data.loc[i,'Daily MAE'] = sum_of_absolute_error/sum_of_numbers

            # mean absolute percentage error
            absolute_percentage_error = (past_data.loc[future_days_index]['Absolute Error']/past_data.loc[future_days_index]['Real'])*100
            data.loc[i,'Daily MAPE'] = absolute_percentage_error.mean()

        data = data.drop([target_name,'date'],axis=1)

    data = data.drop(['prediction datetime','datetime'],axis=1)
    return(data)


#################################################### return data to original mode

def make_original_data(data,target_mode):
    # if target mode is logarithmic we need to return the target variable to its original state
    if target_mode == 'logarithmic':
        data['Target'] = list(np.round(np.exp(data['Target']) - 1))
        data['Prediction'] = list(np.round(np.exp(data['Prediction']) - 1))
    return(data)

################################################### plot


def plot(data):
    data=data[~pd.isna(data['Real'])].drop_duplicates(subset=['the day of the target variable'])
    dates=data['the day of the target variable'].unique().tolist()[40:]
    plot_with = len(dates) + 2
    fig, ax = plt.subplots(figsize=(plot_with, 40))
    # fig, ax = plt.subplots(figsize=(20,5))
    plt.rc('font', size=100)
    plt.plot(dates,data['Real'].tolist()[40:],label='Real number of deaths', linewidth=5.0)
    plt.plot(dates,data['Prediction'].tolist()[40:],label='Predicted number of deaths', linewidth=5.0)
    line_position = dates.index('25 Sep 2020')
    plt.axvline(x=line_position, color='k', linestyle='--')
    plt.ylabel('number of deaths')
    plt.xlabel('date')
    plt.legend()
    plt.xticks(rotation=65)
    locs, labels = plt.xticks()
    weeks = (len(dates)//7)
    plt.xticks([0+(i*7) for i in range(weeks)], np.array(dates)[[0+(i*7) for i in range(weeks)]])
    plt.tight_layout()
    plt.savefig(address + 'US_real_prediction_values.pdf')
    plt.close()


################################################### create tables
def create_tables(y_prediction, y_test_date, y_train_date, r, country_errors, temporal_mode, daily_output,
                  weekly_output, target_mode, run_code, current_date, scenario_flag, futures, covered_r, future_point):

    y_test_date['Prediction'] = list(y_prediction)
    y_test_date = make_original_data(y_test_date,target_mode)
    y_test_date['Prediction'] = best_scaler*y_test_date['Prediction']


    if temporal_mode == 'daily':
      y_train_date['Date'] = y_train_date['date of day t'].apply(
                  lambda x: datetime.datetime.strptime(x, '%y/%m/%d') + datetime.timedelta(days=r))
      y_test_date['Date'] = y_test_date['date of day t'].apply(
                  lambda x: datetime.datetime.strptime(x, '%y/%m/%d') + datetime.timedelta(days=r))
    elif temporal_mode == 'weekly':
      y_train_date['Date'] = y_train_date['date of day t'].apply(
                  lambda x: datetime.datetime.strptime(x, '%y/%m/%d') + datetime.timedelta(days=r*7))
      y_test_date['Date'] = y_test_date['date of day t'].apply(
                  lambda x: datetime.datetime.strptime(x, '%y/%m/%d') + datetime.timedelta(days=r*7))


    ####################################### get country prediction


    y_test_date = y_test_date.reset_index()
    y_test_date['Model'] = best_methods[run_code]
    y_test_date['MASE (test)'] = country_errors['MASE']
    y_test_date['MAPE (test)'] = country_errors['MAPE']
    y_test_date['MAE (test)'] = country_errors['MAE']
    y_test_date['Prediction Date'] = current_date
    y_test_date['run_code'] = run_code
    y_test_date['Absolute Error'] = np.NaN
    y_test_date['Daily MAE'] = np.NaN
    y_test_date['Daily MAPE'] = np.NaN
    y_test_date.rename(columns={'Target':'Real'},inplace=True)
    
    y_test_date = y_test_date[['Prediction Date','Date', 'Real', 'Prediction', 'Model', 'MASE (test)',\
                            'MAPE (test)', 'MAE (test)', 'Absolute Error',\
                            'Daily MAE','Daily MAPE', 'county_fips', 'run_code']]
#     print('y_test_date.columns',y_test_date.columns)

    
    observed = y_test_date.iloc[:-1,:]
#     print('observed\n',observed)
    unobserved = y_test_date.tail(1)
#     print('unobserved\n',unobserved)

    unobserved['Real']=np.NaN 

    if temporal_mode == "daily" :
        observed = date_nominalize(observed,temporal_mode,'Date')
        unobserved = date_nominalize(unobserved,temporal_mode,'Date')
        if first_run==1 :
            daily_output = pd.concat([observed,daily_output], ignore_index=True)
            daily_output = pd.concat([daily_output,unobserved], ignore_index=True)
        else:
            daily_output = pd.concat([daily_output,unobserved.tail(7)], ignore_index=True)
            
#     print('unobserved 727',unobserved['Date'])
    if temporal_mode == "weekly" :
        observed = date_nominalize(observed,temporal_mode,'Date')
        unobserved = date_nominalize(unobserved,temporal_mode,'Date')
        if first_run == 1 and future_point == r-1 : # and r == weekly_r[0]
            weekly_output = pd.concat([observed,weekly_output], ignore_index=True)
            
        weekly_output = pd.concat([weekly_output,unobserved.tail(1)], ignore_index=True)
#     print('unobserved 738',unobserved['Date'])
#     print(y_test_date[['Real']].tail(10), 'line 513')

    return daily_output, weekly_output


def get_current_date(data,temporal_mode):
  if temporal_mode == 'daily':
    data = pd.read_csv(address + 'international-covid-death-data.csv')
    current_date = datetime.datetime.strptime(data.columns[-1],'%m/%d/%y')  #datetime.datetime.strptime('10/18/20','%m/%d/%y') 
  if temporal_mode == 'weekly':
    # data=data[data['date']<=datetime.datetime.strptime('09/25/20','%m/%d/%y') ]
    # data = pd.read_csv(address + 'international-covid-death-data.csv')
    # dates = data.columns
    # for i in range(7):# seven last dates
    #   date = datetime.datetime.strptime(data.columns[-(i+1)],'%m/%d/%y')
    #   if date.weekday()==5:
    #     current_date = date
    #     break
    data['weekday'] = data['date'].apply(lambda x:x.weekday())
    data.sort_values(by=['date','county_fips'],inplace=True)
    data = data.reset_index(drop=True)
    data = data[data['weekday']==5]
    current_date = max(data['date'])
  return(current_date)

############################## reading data

# we use zero_removed data for training the models 
# and then predict target for all counties (set zero_remove argument to 0)
def read_data(h, r, test_size, target_name, target_mode, future_features, pivot, current_date, run_code):
    # current_date = None
    zero_removing = 0
    
    data = futuremakeHistoricalData(h, r, test_size, target_name, 'mrmr', spatial_mode, target_mode, data_address,
                                  future_features, pivot, current_date, zero_removing, run_code)
    data = clean_data(data, numberOfSelectedCounties, spatial_mode)

    if target_mode not in ['regular',
                           'weeklyaverage']:  # we need regular data to return predicted values to first state
        regular_data = futuremakeHistoricalData(h, r, test_size, target_name, 'mrmr', spatial_mode, 'regular', data_address,
                                          future_features, pivot, current_date, zero_removing, run_code)
        regular_data = clean_data(regular_data, numberOfSelectedCounties, spatial_mode)
    else:
        regular_data = data

    return(data,regular_data)

####################### finding errors
    
def find_error(data, regular_data, covar_set, method, r, temporal_mode):
        
    X_train_train, X_train_val, X_test, y_train_train_date, y_train_val_date, y_test_date = preprocess(data, spatial_mode,
                                                                                                        1, r, temporal_mode)
    candid_features = list(X_train_train.columns)
    best_features = candid_features.copy()
    futured_features = ["{}{}".format('future-',i) for i in future_features]
    for feature in candid_features:
        if feature.split(' ')[0] not in covar_set:
            print(feature)
            best_features.remove(feature)
    print(best_features)
    best_features += ['county_fips',
                        'date of day t']
    best_features += force_features
    best_features = np.unique(best_features)

    X_train_train = X_train_train[best_features]
    X_train_val = X_train_val[best_features]


    y_prediction, y_prediction_train = run_algorithms(X_train_train, X_train_val, y_train_train_date, y_train_val_date,
                                                      best_loss, method , 'val')
    county_errors,country_errors,best_scaler = get_errors(y_prediction, y_prediction_train, y_train_val_date, y_train_train_date,
                                                      regular_data, numberOfSelectedCounties, r, temporal_mode)
    
    return(county_errors, country_errors, best_scaler)


############################################# find best configuration for weekly mode

def find_best_configuration(address):
    
    best_r = {r:None for r in range(1,10+1)}
    best_h = {r:None for r in range(1,10+1)}
    best_covariates = {r:[] for r in range(1,10+1)}
    best_methods = {r:None for r in range(1,10+1)}
    best_error = {r:{error:None for error in ['MAPE','MASE','MAE']} for r in range(1,10+1)}
    best_val_error = {r:None for r in range(1,10+1)}
    covered_r = {r:1 for r in range(1,10+1)} # number of r's covered by this r (number of previous weeks which can be predicted using this r)

    for r in range(1,10+1):

        covariates = pd.read_csv(address+str(r)+"/covariates_rank.csv").loc[:,'name'].tolist()

        val_address = address+str(r)+'/0/results/counties=1535/validation/tables/tabel_of_best_validation_results.csv'
        test_address = address+str(r)+'/0/results/counties=1535/test/tables/table_of_best_test_results.csv'
        val_df = pd.read_csv(val_address)
        test_df = pd.read_csv(test_address)
        best_val_error[r] = val_df['percentage of absolute error'].min()
        best_configuration = val_df[val_df['percentage of absolute error']==best_val_error[r]]
        best_h[r] = best_configuration['best_h'].iloc[0]
        best_c = best_configuration['best_c'].iloc[0]
        best_covariates[r] = covariates[:best_c]
        best_methods[r] = best_configuration['method'].iloc[0]
        best_error[r]['MAPE'] = test_df.loc[test_df['method']==best_methods[r],'percentage of absolute error'].iloc[0]
        best_error[r]['MASE'] = test_df.loc[test_df['method']==best_methods[r],'mean absolute scaled error'].iloc[0]
        best_error[r]['MAE'] = test_df.loc[test_df['method']==best_methods[r],'mean absolute error'].iloc[0]
        best_r[r] = r

#         for smaller_r in range(1,r):
#             if best_val_error[r] < best_val_error[smaller_r]:
#                 best_val_error[smaller_r] = best_val_error[r]
#                 best_error[smaller_r] = best_error[r]
#                 best_h[smaller_r] = best_h[r]
#                 best_covariates[smaller_r] = best_covariates[r]
#                 best_methods[smaller_r] = best_methods[r]
#                 best_r[smaller_r] = best_r[r]
#                 covered_r[smaller_r] = 1
#                 covered_r[r] = covered_r[r]+1
    return(best_r,best_methods,best_covariates,best_h,best_error,covered_r)    

############################################### prepare CDC file
def create_CDC_file(first_run):
    if first_run==1:
        data = pd.DataFrame(columns = ['forecast_date','target','target_end_date','location','type','quantile','value'])
    else:
        data = pd.read_csv("CDC_file.csv")
    for r in range(4,10+1):
        temp = pd.read_excel("Weekly-Deaths-Prediction r = "+str(r)+".xlsx")
        temp.rename(columns={'the day the prediction is made':'forecast_date',
                                        'the week of the target variable':'target_end_date', 'Prediction':'value'},inplace = True)
        temp['target'] = np.NaN
        temp['location'] = 'US'
        temp['type'] = 'point'
        temp['quantile'] = 'NA'
        temp['target_end_date'] = temp['target_end_date'].apply(lambda x:datetime.datetime.strptime(x.split(' -- ')[1],"%d %b %Y"))
        temp['forecast_date'] = temp['forecast_date'].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d"))
        temp = temp.tail(1)
        temp = temp.reset_index(drop = True)
        for i in range(len(temp)):
            delta = temp.loc[i,'target_end_date']-temp.loc[i,'forecast_date']
            temp.loc[i,'target'] = str(delta.days//7)+' wk ahead inc death'

        temp['target_end_date'] = temp['target_end_date'].apply(lambda x:datetime.datetime.strftime(x,"%Y-%m-%d"))
        temp['forecast_date'] = temp['forecast_date'].apply(lambda x:datetime.datetime.strftime(x,"%Y-%m-%d"))

        temp = temp[['forecast_date','target','target_end_date','location','type','quantile','value']]
        data = data.append(temp)
    data.to_csv("CDC_file.csv", index=False)

############################################### prepare site file

def create_site_csv(data, temporal_mode, r):
    site_csv = data
    if temporal_mode == 'weekly':
        site_csv = site_csv.rename(columns={'the week of the target variable':'date'})
        site_csv['date'] = site_csv['date'].apply(lambda x:datetime.datetime.strptime(x.split(' -- ')[1],"%d %b %Y").date())
    if temporal_mode == 'daily':
        site_csv = site_csv.rename(columns={'the day of the target variable':'date'})
        site_csv['date'] = site_csv['date'].apply(lambda x:datetime.datetime.strptime(x,"%d %b %Y").date())

    site_csv = site_csv.rename(columns={'Prediction':'predicted_value', 'Real':'real_value', 'Model':'model_number'})
    site_csv['model_number'] = site_csv['model_number'].replace(['KNN','GLM','GBM','NN','MM_GLM','MM_NN'],[1,2,3,4,5,6])

    site_csv['death_or_confirmed'] = 1 if target_name == 'death' else 2

    if country_name == 'US':
        site_csv['country_number'] = 1
    elif country_name == 'Iran':
        site_csv['country_number'] = 2

    if spatial_mode == 'country':
        site_csv['location_level'] = 1
        site_csv['state_number'] = np.NaN
        site_csv['county_number'] = np.NaN
    elif spatial_mode == 'state':
        site_csv['location_level'] = 2
        site_csv = site_csv.rename(columns = {'state_fips':'state_number'})
        site_csv['county_number'] = np.NaN
    elif spatial_mode == 'county':
        site_csv['location_level'] = 3
        site_csv = site_csv.rename(columns = {'county_fips':'county_number'})
        site_csv['state_number'] = site_csv['county_number'].apply(lambda x : int(x//1000))

    site_csv['forecast_horizon_number'] = r
    site_csv = site_csv[['date','predicted_value','real_value','model_number','death_or_confirmed','location_level','country_number','state_number','county_number','forecast_horizon_number']]

    return(site_csv)

################################################## creating site csv for different r's

def create_base_output(first_run):
    if first_run==1:
        data = pd.DataFrame(columns = ['date','predicted_value','real_value','model_number','death_or_confirmed','location_level','country_number','state_number','county_number','forecast_horizon_number'])
    else:
        data = pd.read_csv("site_file_weekly_US.csv")
    for r in range(1,10+1):
        temp = pd.read_csv("site_file_weekly_US r = "+str(r)+".csv")
        
        if first_run==1 and r==1:
            data = data.append(temp)
        else:
            data = data.append(temp.tail(1))
            
    data = data.drop_duplicates(subset = ['date'],keep='last')
    
    data['forecast_horizon_number'] = 0
    data.to_csv('site_file_weekly_US.csv',index = False)
    
########################################################## Reading predicted value of each test_point

def get_test_point_value(address,test_point,r):
    result = pd.read_csv(address+'weeklyaverage/'+str(r)+'/'+str(test_point)+'/results/counties=1535/test/all_errors/KNN/all_errors_KNN.csv')
    test_value = result.iloc[0,:]['prediction']
    return(test_value)

##################################################################################################### main 

def main():

#     get_updated_covid_data(address)
    
    template = pd.DataFrame(columns=['Prediction Date','Date', 'Real', 'Prediction', 'Model', 'MASE (test)',\
                               'MAPE (test)', 'MAE (test)', 'Absolute Error',\
                               'Daily MAE','Daily MAPE', 'run_code','county_fips'])

    daily_output = template 
    weekly_output = template
    
    # calculate number of possible test points for r=1
    base_data = makeHistoricalData(5, 1, 1, target_name, 'mrmr', spatial_mode, "weeklyaverage", data_address, future_features, pivot, 0)
    test_point_limit_r_1 = len(base_data) - 15 # 2 test & val + 13 train ((4/5)*13=10 least number of neighbor)

    covered_r = {r:1 for r in range(1,10+1)}
    for r in range(1,10+1):
        print(200 * '*')
        print('r = ',r)
        
        subprocess.call("python ./prediction.py "+str(r)+" weeklyaverage 0 "+str(argv[1]), shell=True)


    updated_best_r,updated_best_methods,updated_best_covariates,updated_best_h,updated_best_error,covered_r = find_best_configuration(address+"weeklyaverage/")

    run_list = []
    selected_covariates = {}
    
    for r in range(1,10+1):
        run_code = 'w'+str(r*7)
        if updated_best_r[r] == r:
            run_list.append(run_code)
            weekly_r.append(r)
            best_methods[run_code] = updated_best_methods[r]
            best_h[run_code] = updated_best_h[r]
            selected_covariates[run_code] = updated_best_covariates[r]
            
    print('run_list = ',run_list)
    print('best_methods = ',best_methods)
    print('best_h = ',best_h)
    print('updated_best_r = ',updated_best_r)
    print('covered_r = ',covered_r)
    print('selected_covariates = ',selected_covariates)

    for run_code in run_list:
        test_size = 28
        r = int(run_code[1:])
        print ("r = "+str(r))

        if run_code[0] in ['d']:
          temporal_mode="daily"
          target_mode='regular'

        elif run_code[0] in ['w']:
          temporal_mode="weekly"
          target_mode= 'weeklyaverage'
          best_loss = {'GBM': 'poisson', 'MM_NN': 'MeanAbsoluteError', 'NN': 'poisson'}
          r //= 7
          test_size //= 7

        country_errors = updated_best_error[r]
        current_date = get_current_date(temporal_data,temporal_mode)

        h = best_h[run_code]
        print('h = '+str(h))
        
        daily_output = template 
        weekly_output = template
        
        print('current_date = '+ datetime.datetime.strftime(current_date,'%d/%m/%y'))
        

        ####################### reading data and get errors
        
        all_data,all_regular_data = read_data(h, r, test_size, target_name, target_mode, future_features, pivot, current_date, run_code)
        
#         data.to_csv(str(h)+'-'+str(r)+'data.csv',index=False)
#         regular_data.to_csv(str(h)+'-'+str(r)+'regular_data.csv',index=False)
        
        # data = pd.read_csv(str(h)+'-'+str(r)+'data.csv')
        # regular_data = pd.read_csv(str(h)+'-'+str(r)+'regular_data.csv')
        
        for point in range(1,r+1):
            future_point = r-point
            if future_point != 0 :
                data = all_data.copy().iloc[:-future_point,:]
                regular_data = all_regular_data.copy().iloc[:-future_point,:]
            else:
                data = all_data.copy()
                regular_data = all_regular_data.copy()
            print('r=',r)
            print('future_point=',future_point)
            print('data=',len(data))
            if temporal_mode == 'daily':
                county_errors, country_errors, best_scaler = find_error(data, regular_data, selected_covariates[run_code],
                                                                        best_methods[run_code], r, temporal_mode)
            if temporal_mode == 'weekly':
                county_errors = updated_best_error[r]

            ####################### prediction future

            X_train, _, y_train_date, _ = preprocess(data,spatial_mode,0, r, temporal_mode, 1)

            # X_train, _, _, y_train_date, _, _ = preprocess(data,spatial_mode,1, r, temporal_mode, 1)

            X_X_train, X_test , y_y_train_date, y_test_date = preprocess(data,spatial_mode,0, r, temporal_mode, 0)

            # consider all data as test to predicting observed targets beside unobserved targets
            X_test = X_X_train.append(X_test)
            y_test_date = y_y_train_date.append(y_test_date)


            candid_features = list(X_train.columns)
            best_features = candid_features.copy()
            for feature in candid_features:
                if feature.split(' ')[0] not in selected_covariates[run_code]:
                    best_features.remove(feature)
            best_features += ['county_fips',
                                'date of day t'] 
            best_features = np.unique(best_features)


            X_train = X_train[best_features]
            X_test = X_test[best_features]
            X_train.to_csv('X_train.csv')
            y_prediction, y_prediction_train = run_algorithms(X_train, X_test, y_train_date, y_test_date,
                                                          best_loss, best_methods[run_code], 'test')


            if run_code == 'd14':
                daily_output, _ = create_tables(y_prediction, y_test_date, y_train_date, r, country_errors, temporal_mode,
                                                            daily_output, weekly_output, target_mode, run_code, current_date,
                                                scenario_flag = 0, futures = selected_futures, covered_r = covered_r,future_point=future_point)

            elif run_code[0] == 'w':

                _, weekly_output = create_tables(y_prediction, y_test_date, y_train_date, r, country_errors, temporal_mode,
                                                     daily_output, weekly_output, target_mode, run_code, current_date,
                                                 scenario_flag = 0, futures = selected_futures, covered_r = covered_r,future_point=future_point)

        weekly_output['Prediction Date']=weekly_output['Prediction Date'].apply(lambda x: datetime.datetime.strftime(x,'%Y-%m-%d'))
        daily_output['Prediction Date']=daily_output['Prediction Date'].apply(lambda x: datetime.datetime.strftime(x,'%Y-%m-%d'))
    
        if first_run==0: 
            weekly_output_csv = pd.read_csv("weekly_backup r = "+str(r)+".csv")
            weekly_output_csv = weekly_output_csv.append(weekly_output)

            daily_output_csv = pd.read_csv("daily_backup.csv")
            daily_output_csv = daily_output_csv.append(daily_output)


        else:
            weekly_output_csv = weekly_output
            daily_output_csv = daily_output


        # add real values of previous days
        weekly_output_csv = add_real_values(weekly_output_csv,address,'weekly')
        daily_output_csv = add_real_values(daily_output_csv,address,'daily')


        # remove duplaicates
        for col in ['MASE (test)', 'MAPE (test)', 'MAE (test)']:
               daily_output_csv.loc[daily_output_csv[col].duplicated(), col]=np.nan
               weekly_output_csv.loc[weekly_output_csv[col].duplicated(), col]=np.nan

        daily_output_csv.to_csv('daily_backup.csv',index = False)
        weekly_output_csv.to_csv('weekly_backup r = '+str(r)+'.csv',index = False)


        ############################## prepare output csv file

        daily_output_csv=daily_output_csv.drop(['run_code'],axis=1)
        weekly_output_csv=weekly_output_csv.drop(['run_code'],axis=1)


        daily_output_csv['Absolute Error'] = abs(daily_output_csv['Real']-daily_output_csv['Prediction'])
        weekly_output_csv['Absolute Error'] = abs(weekly_output_csv['Real']-weekly_output_csv['Prediction'])

        numerical_cols = ['Real', 'Prediction','Absolute Error', 'MASE (test)',\
                                       'MAPE (test)', 'MAE (test)', 
                                       'Daily MAE', 'Daily MAPE']

        for col in numerical_cols:
          daily_output_csv[col]=daily_output_csv[col].apply(lambda x : np.round(x,2))
          weekly_output_csv[col]=weekly_output_csv[col].apply(lambda x : np.round(x,2))

        ordered_cols = ['Prediction Date','Date', 'Real', 'Prediction','Absolute Error', 'Model', 'MASE (test)',\
          'MAPE (test)', 'MAE (test)', 'Daily MAE', 'Daily MAPE']     

        daily_output_csv = daily_output_csv[ordered_cols]
        weekly_output_csv = weekly_output_csv[ordered_cols]

        daily_output_csv = daily_output_csv.rename(columns={'Date':'the day of the target variable', 'Prediction Date':'the day the prediction is made',\
                                     'Absolute Error':'difference'})
        weekly_output_csv = weekly_output_csv.rename(columns={'Date':'the week of the target variable', 'Prediction Date':'the day the prediction is made',\
                                     'Absolute Error':'difference', 'Daily MAE':'Weekly MAE','Daily MAPE':'Weekly MAPE'})

        # save plot of real and predicted values
        # plot(daily_output_csv)

        daily_output_csv.to_excel("Daily-Deaths-Prediction.xlsx", index=False)
        weekly_output_csv.to_excel("Weekly-Deaths-Prediction r = "+str(r)+".xlsx", index=False)

        weekly_output_csv = weekly_output_csv.drop_duplicates(subset = ['the week of the target variable'],keep='last')
        daily_output_csv = daily_output_csv.drop_duplicates(subset = ['the day of the target variable'],keep='last')
    
        daily_site_csv = create_site_csv(daily_output_csv, 'daily', r)
        daily_site_csv.to_csv("site_file_daily_US.csv", index=False)

        weekly_site_csv = create_site_csv(weekly_output_csv, 'weekly', r)
        weekly_site_csv.to_csv("site_file_weekly_US r = "+str(r)+".csv", index=False)
        
    create_CDC_file(first_run)
    create_base_output(first_run)

if __name__ == "__main__":
    
    main()
