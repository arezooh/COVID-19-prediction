#!/usr/bin/python3
import pandas as pd
import numpy as np
import requests
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
import datetime
import os
from zipfile import ZipFile
from sys import argv

# self imports
import debug
import handlers
import extractor
import medium

csv_address = './csvFiles/'

first_run = 1
weather_flag = 1 # decide for downloading weather data or not

save_address = csv_address + 'all_features_data/'
if weather_flag == 0:
    save_address = csv_address + 'weatherless/'
if not os.path.exists(save_address):
    os.makedirs(save_address)

def get_csv(web_addres,file_address):
    url=web_addres
    print(url)
    req = requests.get(url)
    url_content = req.content
    csv_file = open(file_address, 'wb')
    csv_file.write(url_content)
    csv_file.close
    
def get_zip(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

if __name__ == "__main__":
    
#     if weather_flag:
        
#         if os.path.exists("./weather.log"):
#             os.remove("./weather.log")
#         else:
#             print("The file does not exist")
        
    # downloadHandler.get_socialDistancingData(2, 'sd-state%02d.json' % (2))
    # downloadHandler.get_confirmAndDeathData( + 'confirmAndDeath.json')
    # csvHandler.simplify_csvFile(csv_address+'latitude.csv', csv_address+'simple-lat.csv', ['county_fips', 'lat'])
    # csvHandler.simplify_csvFile(csv_address+'hospital-beds.csv', csv_address+'simple-hospital-beds.csv', ['county_fips', 'beds(per1000)', 'unoccupiedBeds(per1000)'])
    # csvHandler.merge_csvFiles_addRows(csv_address+'socialDistancing-s01.csv', csv_address+'socialDistancing-s11.csv', csv_address+'socialDistancing.csv')
    # jsonHandler.transform_jsonToCsv_confirmAndDeathData( + 'confirmAndDeath.json',  + 'confirmAndDeath.csv')
    # jsonHandler.transform_jsonToCsv_hospitalBedData( + 'hospital-beds.json',  + 'hospital-beds.csv')
    # jsonHandler.transform_jsonToCsv_socialDistancingData( + 'sd-state01.json',  + 'socialDistancing-s01.csv')
    # jsonHandler.transform_jsonToCsv_socialDistancingData('sd-state02.json',  + 'socialDistancing-s02.csv')
    
    # get Social Distancing data
    if first_run:
        
        mediumObject = medium.mediumClass()
        mediumObject.generate_allSocialDistancingData()

    # jsonHandler.transform_jsonToCsv_confirmAndDeathData('confirmAndDeath.json', 'temp-confirmAndDeath.csv')
    # downloadHandler.get_allStations('stations.csv')
    # downloadHandler.get_countyWeatherData('USW00093228', '2020-04-19', '2020-04-28', 'test.csv') #https://www.ncdc.noaa.gov/cdo-web/datasets/GHCND/stations/GHCND:USW00093228/detail


    #       |--Use these two lines to get all stations--|
    # mediumObject = medium.mediumClass()
    # mediumObject.downloadHandler.get_allStations('stations.csv')
    #       |--|


    #       |--Use this line to test merging two temporalDataFile: 'confirmAndDeath.csv' and 'socialDistancing.csv'--|
    # mediumObject = medium.mediumClass()
    # mediumObject.csvHandler.merge_csvFiles_addColumns('confirmAndDeath.csv', 'socialDistancing.csv', 'temporal-data.csv', ['countyFIPS', 'date'], ['countyFips', 'date'], ['totalGrade', 'visitationGrade', 'encountersGrade', 'travelDistanceGrade'])
    #       |--|
    
#     if weather_flag :
#         weather=pd.read_csv(csv_address+'weather.csv')
#         weather=weather.dropna(subset=['DATE'])
#         weather['DATE']=weather['DATE'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
#         startdate = datetime.datetime.strftime(max(weather['DATE'] - datetime.timedelta(days=10)) ,'%Y-%m-%d')
#         print('weather start date: ',startdate)
#         today = datetime.datetime.now()
#         enddate = datetime.datetime.strftime(today ,'%Y-%m-%d')
#         print('weather end date: ',enddate)
#         stations = pd.read_csv(csv_address+'stations.csv')
#         print ('stations.shape',stations.shape)

#         if first_run:
#             new_weather = pd.DataFrame(columns=weather.columns)
#             new_weather.to_csv(csv_address+'new-weather.csv',index=False)
#             stations.to_csv(csv_address+'temp-stations.csv')
#         else:
#             weather=pd.read_csv(csv_address+'new-weather.csv')
#             stations2=stations.copy()
#             stations2['id']=stations2['id'].apply(lambda x: x[6:])
#             ind=stations2[~(stations2['county_fips'].isin(weather['county_fips'].unique()[:-1]))].index
#             stations=stations.iloc[ind,:]
#             stations.to_csv(csv_address+'temp-stations.csv', index=False)

#         # get weather data

#         mediumObject = medium.mediumClass()
#         mediumObject.downloadHandler.get_countyWeatherData('1001', 'USW00093228', startdate, enddate, 'test.csv')
#         mediumObject.generate_allWeatherData(startdate, enddate)

    # mediumObject = medium.mediumClass()
    # mediumObject.downloadHandler.get_airlines()
    
#     # get confirmed cases data
#     get_csv('https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_confirmed_usafacts.csv',\
#         csv_address+'covid_confirmed_cases.csv')
#     # get deaths data
#     get_csv('https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_deaths_usafacts.csv',\
#             csv_address+'covid_deaths.csv')
#     # get tests data
#     get_csv('https://covidtracking.com/api/v1/states/daily.csv',\
#             csv_address+'new-daily-state-test.csv')
#     # get region google mobility data
#     get_zip('https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip',\
#             csv_address+'Region_Mobility_Report_CSVs.zip')
#     # get global google mobility data
#     get_csv('https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv',csv_address+'Global_Mobility_Report.csv')

#     # get international confirmed cases and death data
#     get_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
#     ,csv_address + 'international-covid-death-data.csv')
#     get_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
#     ,csv_address + 'international-covid-confirmed-data.csv')
    ########################### add new weather to weather file
#     if weather_flag :
#         new_weather=pd.read_csv(csv_address+'new-weather.csv')
#         weather=pd.read_csv(csv_address+'weather.csv')
#         weather=weather.append(new_weather)
#         weather=weather.drop_duplicates(subset=['county_fips','STATION','DATE'])
#         weather.to_csv(csv_address+'weather.csv', index=False)
    
    ########################### add new tests to test file
    new_tests=pd.read_csv(csv_address+'new-daily-state-test.csv')
    tests=pd.read_csv(csv_address+'daily-state-test.csv')
    tests=tests.append(new_tests)
    tests=tests.drop_duplicates(subset=['date','state'])
    tests.to_csv(csv_address+'daily-state-test.csv', index=False)
    
    ########################################################################## concat and prepare data
    
    fix=pd.read_csv(csv_address+'fixed-data.csv')
    socialDistancing=pd.read_csv(csv_address+'socialDistancing.csv')
    cof=pd.read_csv(csv_address+'covid_confirmed_cases.csv')
    
    zipFileName = csv_address+'Region_Mobility_Report_CSVs.zip'
    US_address = '2020_US_Region_Mobility_Report.csv'
    with ZipFile(zipFileName, 'r') as zip:
        US_file = zip.extract(US_address)
    google_mobility_data = pd.read_csv(US_file)    

    # max date recorded
    confirmed_and_death_max_date = max([datetime.datetime.strptime(x,'%m/%d/%y') for x in cof.columns[4:]]).date()

    # preprocess socialDistancing
    socialDistancing=socialDistancing[['countyFips', 'date',
           'totalGrade', 'visitationGrade', 'encountersGrade',
           'travelDistanceGrade']]
    socialDistancing=socialDistancing.rename(columns={'countyFips':'county_fips'})
    socialDistancing['county_fips']=socialDistancing['county_fips'].apply(lambda x:x[2:7]).astype(int)
    socialDistancing['date']=socialDistancing['date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))

    # max date recorded
    socialDistancing_max_date = max(socialDistancing['date']).date()
    
    
    # find dates which confirmed cases and deaths are recorded
    valid_dates=cof.columns[4:].tolist()

    ################################################################### create template for data

    fips=pd.DataFrame(columns=['fips'])
    fips['fips']=fix['county_fips'].tolist()*len(valid_dates)
    fips.sort_values(by='fips',inplace=True)
    data=pd.DataFrame(columns=['county_fips','date'])
    data['county_fips']=fips['fips']
    data['date']=valid_dates*3142 
    data['date']=data['date'].apply(lambda x: datetime.datetime.strptime(x,'%m/%d/%y'))
    data.sort_values(by=['county_fips','date'],inplace=True)

    ################################################################### add socialDistancing data

    data=pd.merge(data,socialDistancing,how='left',left_on=['county_fips','date'],right_on=['county_fips','date'])
    data=data.rename(columns={'totalGrade':'social-distancing-total-grade','visitationGrade':'social-distancing-visitation-grade',
                'encountersGrade':'social-distancing-encounters-grade','travelDistanceGrade':'social-distancing-travel-distance-grade'})

    data['social-distancing-total-grade']=data['social-distancing-total-grade'].replace(['A','A-','B+','B','B-','C+','C','C-','D+','D','D-','F']\
                                                                                        ,[12,11,10,9,8,7,6,5,4,3,2,1])

    for i in ['social-distancing-visitation-grade','social-distancing-encounters-grade',\
              'social-distancing-travel-distance-grade']:
        data[i]=data[i].replace(['A','B','C','D','F'],[5,4,3,2,1])
        
    ################################################################## add google-mobility data
    
    # preprocess google mobility
    google_mobility_data = google_mobility_data[~pd.isnull(google_mobility_data['census_fips_code'])]
    google_mobility_data = google_mobility_data.rename(columns={
           'census_fips_code':'county_fips', 'retail_and_recreation_percent_change_from_baseline':'Retail',
           'grocery_and_pharmacy_percent_change_from_baseline':'Grocery',
           'parks_percent_change_from_baseline': 'Parks',
           'transit_stations_percent_change_from_baseline':'Transit',
           'workplaces_percent_change_from_baseline':'Workplace',
           'residential_percent_change_from_baseline':'Residential'})
    google_mobility_data = google_mobility_data[['county_fips', 'date', 'Retail', 'Grocery', 'Parks', 'Transit', 'Workplace', 'Residential']]
    google_mobility_data['date'] = google_mobility_data['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    
    data=pd.merge(data,google_mobility_data,how='left',left_on=['county_fips','date'],right_on=['county_fips','date'])
    google_mobility_max_date = max(google_mobility_data['date']).date()
    

    ################################################################## add week-day
    data['week-day']=data['date']#
    data['week-day']=data['week-day'].apply(lambda x:x.weekday())

    data['week-day']=data['week-day'].replace([0,1,2,3,4,5,6],[0,0,0,0,1,2,1])

    data.rename(columns={'week-day':'weekend'},inplace=True)

    ################################################################ add test

    dailytest = pd.read_csv(csv_address+'daily-state-test.csv')
    dailytest['date']=dailytest['date'].astype(str).apply(lambda x:datetime.datetime.strptime(x,'%Y%m%d'))
    dailytest=dailytest[['date','fips','totalTestResultsIncrease']]

    test_max_date = max(dailytest['date']).date()

    data['fips']=data['county_fips']//1000
    data=pd.merge(data,dailytest,how='left',left_on=['date','fips'],right_on=['date','fips'])

    state_pop=fix[['state_fips','total_population']].groupby(['state_fips']).sum()
    state_pop=state_pop.reset_index()

    data=pd.merge(data,state_pop,how='left',left_on=['fips'],right_on=['state_fips'])

    data['totalTestResultsIncrease']=data['totalTestResultsIncrease']/data['total_population']

    data.drop(['fips','state_fips','total_population'],axis=1,inplace=True)
    data.rename(columns={'totalTestResultsIncrease':'daily-state-test'},inplace=True)

    ############################################################## add weather data

    if weather_flag :
        
        weather=pd.read_csv(csv_address+'weather.csv',na_values=np.nan)

        weather=weather.dropna(subset=['DATE'])
        weather['DATE']=weather['DATE'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))

        weather_max_date = max(weather['DATE'].unique())
        if isinstance(weather_max_date,np.datetime64):
            weather_max_date = datetime.datetime.utcfromtimestamp(weather_max_date.tolist()/1e9).date()
        elif isinstance(weather_max_date,pd._libs.tslibs.timestamps.Timestamp):
            weather_max_date = weather_max_date.date

        perc=weather.dropna(subset=['PRCP'])[['county_fips','DATE','PRCP']]
        perc.drop_duplicates(subset=['county_fips','DATE'],inplace=True)
        data=pd.merge(data,perc,how='left',left_on=['county_fips','date'],right_on=['county_fips','DATE'])
        data.drop(['DATE'],axis=1,inplace=True)
        data.rename(columns={'PRCP':'precipitation'},inplace=True)

        # impute average using min and max values
        rows_with_null_average=weather[(pd.isnull(weather['TAVG']))&(~pd.isnull(weather['TMAX']))&(~pd.isnull(weather['TMIN']))].index.tolist()
        weather.loc[rows_with_null_average,'TAVG']=weather.loc[rows_with_null_average,'TMAX']+weather.loc[rows_with_null_average,'TMIN']
        weather.loc[rows_with_null_average,'TAVG']=weather.loc[rows_with_null_average,'TAVG']/2
        weather.loc[rows_with_null_average,'TAVG']=weather.loc[rows_with_null_average,'TAVG'].round()

        temperature=weather.copy()[['county_fips', 'DATE','TAVG']]
        temperature.dropna(subset=['TAVG'],inplace=True)
        temperature.drop_duplicates(subset=['county_fips','DATE'],inplace=True)
        data=pd.merge(data,temperature,how='left',left_on=['county_fips','date'],right_on=['county_fips','DATE'])
        data.drop(['DATE'],axis=1,inplace=True)
        data.rename(columns={'TAVG':'temperature'},inplace=True)
        # recorrect scale
        data['temperature']=data['temperature']/10

    ############################################################# add confirmed cases and deaths

    cof=pd.read_csv(csv_address+'covid_confirmed_cases.csv')
    det=pd.read_csv(csv_address+'covid_deaths.csv')

    # derive new cases from cumulative cases
    cof2=cof.copy()
    for i in range(5,cof.shape[1]):
        cof2.iloc[:,i]=cof.iloc[:,i]-cof.iloc[:,i-1]
    cof=cof2.copy()

    det2=det.copy()
    for i in range(5,det.shape[1]):
        det2.iloc[:,i]=det.iloc[:,i]-det.iloc[:,i-1]
    det=det2.copy()

    cof=cof[cof['countyFIPS'].isin(data['county_fips'])]
    det=det[det['countyFIPS'].isin(data['county_fips'])]

    # add new cases to data
    data=data.drop_duplicates(subset=['county_fips','date'])

    for i in cof.columns[4:cof.shape[1]]:
        j=datetime.datetime.strptime(i,'%m/%d/%y')
        data.loc[data['date']==j,'confirmed']=cof[i].copy().tolist()

    for i in det.columns[4:cof.shape[1]]:
        j=datetime.datetime.strptime(i,'%m/%d/%y')
        data.loc[data['date']==j,'death']=det[i].copy().tolist()

    # save unimputed data
    unimputed_data=data.copy()

    # impute negative number of confirmed cases and deaths

    reverse_dates=cof.columns[4:][::-1]
    while cof.iloc[:,4:].min().sum()<0:
        for i in range(len(reverse_dates)):
            date = reverse_dates[i]
            negative_index=cof[cof[date]<0].index
            cof2.loc[negative_index,date] = 0
        for i in range(len(reverse_dates)):
            date = reverse_dates[i]
            past_date = reverse_dates[i+1]
            negative_index=cof[cof[date]<0].index
            cof2.loc[negative_index,past_date] = cof2.loc[negative_index,past_date]+cof.loc[negative_index,date]
            if i==len(reverse_dates)-2:
                break
        cof=cof2.copy()

    reverse_dates=det.columns[4:][::-1]
    while det.iloc[:,4:].min().sum()<0:
        for i in range(len(reverse_dates)):
            date = reverse_dates[i]
            negative_index=det[det[date]<0].index
            det2.loc[negative_index,date] = 0
        for i in range(len(reverse_dates)):
            date = reverse_dates[i]
            past_date = reverse_dates[i+1]
            negative_index=det[det[date]<0].index
            det2.loc[negative_index,past_date] = det2.loc[negative_index,past_date]+det.loc[negative_index,date]
            if i==len(reverse_dates)-2:
                break
        det=det2.copy()

    cof=cof[cof['countyFIPS'].isin(data['county_fips'])]
    det=det[det['countyFIPS'].isin(data['county_fips'])]

    # add imputed values to data

    for i in cof.columns[4:cof.shape[1]]:
        j=datetime.datetime.strptime(i,'%m/%d/%y')
        data.loc[data['date']==j,'confirmed']=cof[i].copy().tolist()

    for i in det.columns[4:cof.shape[1]]:
        j=datetime.datetime.strptime(i,'%m/%d/%y')
        data.loc[data['date']==j,'death']=det[i].copy().tolist()

    ########################################################################## add virus pressure to data

    adjacency_matrix=pd.read_csv(csv_address+'adj_mat.csv')
    adjacency_matrix.index=adjacency_matrix['Unnamed: 0']
    adjacency_matrix.drop('Unnamed: 0',axis=1,inplace=True)

    data['virus-pressure']=0

    confirmed=pd.DataFrame(index=data['county_fips'].unique(),columns=data['date'].unique())

    for i in confirmed.columns:
        confirmed[i]=data.loc[data['date']==i,'confirmed'].tolist()
    for i in confirmed.columns:
        confirmed[i]=confirmed[i]
    for i in data['date'].unique():
        data.loc[data['date']==i,'virus-pressure']=np.dot(adjacency_matrix,confirmed[[i]])

    adjacency_matrix['neighbur_count'] = adjacency_matrix.sum().values.tolist()

    adjacency_matrix['county_fips']=adjacency_matrix.index
    adjacency_matrix.loc[adjacency_matrix['neighbur_count']==0,'neighbur_count']=1
    data=pd.merge(data,adjacency_matrix[['neighbur_count','county_fips']])
    data['virus-pressure']=data['virus-pressure']/data['neighbur_count']

    data=data.drop(['neighbur_count'],axis=1)

    unimputed_data['virus-pressure']=data['virus-pressure']#####**************************************######

    ###################################################################### 
    # find max date with all features recorded and save unimputed data

    max_date = min(socialDistancing_max_date,confirmed_and_death_max_date,test_max_date,google_mobility_max_date)
    
    if weather_flag:
        max_date = min(weather_max_date, max_date)
    
    max_date = datetime.datetime.combine(max_date, datetime.datetime.min.time())

    data=data[data['date']<=max_date]
    unimputed_data=unimputed_data[unimputed_data['date']<=max_date]

    #data['date']=data['date'].apply(lambda x: x.strftime('%m/%d/%y'))
    unimputed_data['date']=unimputed_data['date'].apply(lambda x: x.strftime('%y/%m/%d'))
    unimputed_data.to_csv(save_address+'unimputed-temporal-data.csv',index=False)

    ########################################################################## imputation

    #data['date']=data['date'].apply(lambda x:datetime.datetime.strptime(x,'%m/%d/%y'))

    covariate_to_imputed = ['social-distancing-total-grade','social-distancing-visitation-grade',
                            'social-distancing-encounters-grade','social-distancing-travel-distance-grade',
                            'Retail', 'Grocery', 'Parks', 'Transit', 'Workplace', 'Residential']
    if weather_flag:
        covariate_to_imputed = covariate_to_imputed + ['precipitation','temperature']

    # save counties with all nulls
    temp=data.groupby('county_fips').count()
    counties_with_all_null_ind={}
    for i in covariate_to_imputed:
        counties_with_all_null_ind[i]=temp[temp[i]==0].index.tolist()


    # impute first days social distancing with lowest value    
    social_distancing_grades = ['social-distancing-total-grade','social-distancing-visitation-grade',
                                'social-distancing-encounters-grade','social-distancing-travel-distance-grade']

    for social_distancing_grade in social_distancing_grades:
        data.loc[(data['date']<datetime.datetime(2020,2,24)),social_distancing_grade]=1 # we have no social distancing data before 2020,2,24

    google_mobility_features = ['Retail', 'Grocery', 'Parks', 'Transit', 'Workplace', 'Residential']
    
    for feature in google_mobility_features:
        data.loc[(data['date']<datetime.datetime(2020,2,15)),feature]=0
        
    
    data['date']=data['date'].apply(lambda x: x.strftime('%y/%m/%d'))


    # impute covariates with KNN imputer

    for covar in covariate_to_imputed:

        print(covar,' null count:',len(counties_with_all_null_ind[covar]))

        temp=pd.DataFrame(index=data['county_fips'].unique().tolist(),columns=data['date'].unique().tolist())

        for i in data['date'].unique():
            temp[i]=data.loc[data['date']==i,covar].tolist()
        if covar in social_distancing_grades: # data is not recorded in this day
            temp.loc[pd.isna(temp['21/01/12']),'21/01/12']=1

        X = np.array(temp)
        imputer = KNNImputer(n_neighbors=5)
        imp=imputer.fit_transform(X)
        imp=pd.DataFrame(imp)
        imp.columns=temp.columns
        imp.index=temp.index
        for i in data['date'].unique():
            data.loc[data['date']==i,covar]=imp[i].tolist()

        # delete values for all null counties
        data.loc[data['county_fips'].isin(counties_with_all_null_ind[covar]),covar]=np.NaN

    # remove features with high volume of missing from imputed data 
    data=data.drop(['social-distancing-visitation-grade', 'Parks', 'Transit', 'Residential'],axis=1)

    # impute state daily test

    first_day_null = data.loc[pd.isnull(data['daily-state-test'])].index

    data.loc[data['daily-state-test']<0,'daily-state-test']=np.NaN
    value_count=data.groupby('county_fips').count()
    counties_with_all_nulls=value_count[value_count['daily-state-test']==0]
    temp=pd.DataFrame(index=data['county_fips'].unique().tolist(),columns=data['date'].unique().tolist())

    for i in data['date'].unique():
        temp[i]=data.loc[data['date']==i,'daily-state-test'].tolist()
    X = np.array(temp)
    imputer = KNNImputer(n_neighbors=5)
    imp=imputer.fit_transform(X)
    imp=pd.DataFrame(imp)
    imp.columns=temp.columns
    imp.index=temp.index
    for i in data['date'].unique():
        data.loc[data['date']==i,'daily-state-test']=imp[i].tolist()
    if(len(counties_with_all_nulls)>0):
        data.loc[data['county_fips'].isin(counties_with_all_nulls.index),'daily-state-test']=np.NaN

    #####**************************************######
    data2=data.copy()
    data2.loc[first_day_null,'daily-state-test']=np.NaN # return first days nulls

    # save data for models
    data2=data2.sort_values(by=['county_fips','date'])
    data2.to_csv(save_address+'temporal-data.csv',index=False)

    #################################################### remove counties with all nulls for some features

    fixed_features_with_nulls=['ventilator_capacity','icu_beds','deaths_per_100000']

    for i in fixed_features_with_nulls:
        nullind=fix.loc[pd.isnull(fix[i]),'county_fips'].unique()
        print(len(nullind))
        data=data[~data['county_fips'].isin(nullind)]
        fix=fix[~fix['county_fips'].isin(nullind)]

    timeDeapandant_features_with_nulls=['social-distancing-travel-distance-grade','social-distancing-total-grade',
                                        'social-distancing-encounters-grade', 'Retail', 'Grocery',
                                        'Workplace']
    if weather_flag:
        timeDeapandant_features_with_nulls=timeDeapandant_features_with_nulls+['temperature','precipitation']

    for i in timeDeapandant_features_with_nulls:
        nullind=data.loc[pd.isnull(data[i]),'county_fips'].unique()
        print(len(nullind))
        data=data[~data['county_fips'].isin(nullind)]
        fix=fix[~fix['county_fips'].isin(nullind)]


    ##### saivng final imputed data 
    fix.to_csv(save_address+'full-fixed-data.csv', index=False)
    data.to_csv(save_address+'full-temporal-data.csv', index=False)

    fix[['county_fips']].to_csv(save_address+'full-data-county-fips.csv',index=False)
