from sys import argv
import sys
import os
import subprocess
from pexecute.process import ProcessLoom
import requests
import pandas as pd
import datetime


weather_stop = int(argv[1])

########################################################### push results to github
def push(message):
    
##    subprocess.run("git pull", check=True, shell=True)
##    print("everything has been pulled")
##    subprocess.run("git add .", check=True, shell=True)
##    subprocess.call(["git", "commit", "-m", "'{}'".format(message), "--allow-empty"], shell=True)
##    subprocess.run("git push", check=True, shell=True)
    print('pushed.')
    
def update_status(country):
    
    if country == 'usa':
        prediction = pd.read_csv("./"+ country +"/site_file_weekly_US r = 1.csv")
        prediction['date'] = prediction['date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
        weatherless_data = pd.read_csv("./csvFiles/weatherless/temporal-data.csv")
        weatherless_data['date'] = weatherless_data['date'].apply(lambda x: datetime.datetime.strptime(x,'%y/%m/%d'))
        if max(weatherless_data['date']) >= max(prediction['date']) :
            return(True)
        
    elif country == 'iran':
        prediction = pd.read_csv("./"+ country +"/site_file_weekly_Iran r = 1.csv")
        prediction['date'] = prediction['date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
        updated_covid_data = pd.read_csv("./csvFiles/international-covid-death-data.csv")
        last_date = datetime.datetime.strptime(updated_covid_data.columns[-1],'%m/%d/%y')
        if (last_date) >= max(prediction['date']) :
            return(True)
        
    elif country == 'canada':
        prediction = pd.read_csv("./"+ country +"/site_file_weekly_Canada r = 1.csv")
        prediction['date'] = prediction['date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
        updated_covid_data = pd.read_csv("./csvFiles/international-covid-death-data.csv")
        last_covid_date = datetime.datetime.strptime(updated_covid_data.columns[-1],'%m/%d/%y')
        updated_mobility_data = pd.read_csv("./csvFiles/Global_Mobility_Report.csv")
        updated_mobility_data['date'] = updated_mobility_data['date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
        last_mobility_date = max(updated_mobility_data['date'])
        last_date = min(last_mobility_date,last_covid_date)
        if (last_date) >= max(prediction['date']) :
            return(True)
        
    return(False)
    
def main():
     # if weather stop is 0 its mean this is the first run of the main procedure but if weather stop is 1
     # its mean that downloading weather data is interrupted in the middle of the run and we must run the main
     # procedure again so we ignore the parts which are run before downloading weather data 
##    if weather_stop == 0:
        subprocess.call("python ./update_data/update_data.py f 0", shell=True)
        print('Data updated without weather')
        
##        if update_status('iran'):
##            subprocess.call("python ./iran/future_prediction.py", shell=True)
##            print('Iran prediction is updated')
##        if update_status('canada'):
##            subprocess.call("python ./canada/future_prediction.py", shell=True)
##            print('canada prediction is updated')
##        push("Predictions are updated")

    
##    if update_status('usa') or (weather_stop == 1):
##        if weather_stop == 0: # first run of weather download
####            subprocess.call("python ./usa/future_prediction.py 0", shell=True)
####            print('USA prediction is updated')
####            push("US prediction is updated")
##            subprocess.call("python ./update_data/update_data.py f 1", shell=True)
##            print('Data updated with weather')
##        else:
##            subprocess.call("python ./update_data/update_data.py s 1", shell=True)
##            print('Data updated with weather')
##        weatherless_data = pd.read_csv("./csvFiles/weatherless/temporal-data.csv")
##        data = pd.read_csv("./csvFiles/temporal-data.csv")
##        weatherless_data['date'] = weatherless_data['date'].apply(lambda x: datetime.datetime.strptime(x,'%y/%m/%d'))
##        data['date'] = data['date'].apply(lambda x: datetime.datetime.strptime(x,'%y/%m/%d'))
##        if max(data['date'])>=max(weatherless_data['date']):
####            subprocess.call("python ./usa/future_prediction.py #1", shell=True)
##            print('USA prediction using data includes weather is updated')
####            push("US predictions updated using weather+ data")
##            subprocess.call("python ./update_data/all_features_update_data.py", shell=True) # update data with all features
        
        
if __name__ == "__main__":
    main()
