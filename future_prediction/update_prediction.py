from sys import argv
import sys
import os
import subprocess
from pexecute.process import ProcessLoom
import requests
import pandas as pd

########################################################### push results to github
def push(message):
    
    subprocess.run("git pull", check=True, shell=True)
    print("everything has been pulled")
    subprocess.run("git add .", check=True, shell=True)
    subprocess.call(["git", "commit", "-m", "'{}'".format(message), "--allow-empty"], shell=True)
    subprocess.run("git push", check=True, shell=True)
    print('pushed.')
    
def main():
    subprocess.call("python ./update_data/update_data.py 0", shell=True)
    print('Data updated without weather')
    subprocess.call("python ./iran/future_prediction.py", shell=True)
    print('Iran prediction is updated')
    subprocess.call("python ./canada/future_prediction.py", shell=True)
    print('canada prediction is updated')
    subprocess.call("python ./usa/future_prediction.py 0", shell=True)
    print('USA prediction is updated')
    push("Predictions updated")
    subprocess.call("python ./update_data/update_data.py 1", shell=True)
    print('Data updated with weather')
    weatherless_data = pd.read_csv("../csvFiles/weatherless/temporal-data.csv")
    data = pd.read_csv("../csvFiles/temporal-data.csv")
    weatherless_data['date'] = weatherless_data['date'].apply(lambda x: datetime.datetime.strptime(x,'%y/%m/%d'))
    data['date'] = data['date'].apply(lambda x: datetime.datetime.strptime(x,'%y/%m/%d'))
    if max(data['date'])>=max(weatherless_data['date']):
        subprocess.call("python ./usa/future_prediction.py 1", shell=True)
        print('USA prediction using data includes weather is updated')
        push("Predictions updated")

if __name__ == "__main__":
    main()
