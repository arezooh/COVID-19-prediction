from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
import pandas as pd
import numpy as np

def main():
    r = 5 * 7
    template = pd.DataFrame(columns = ['Unnamed: 0','date of day t','county_fips','Target','prediction','error','absoulte_error','percentage_error'])
    methods = ['GBM', 'GLM', 'KNN', 'NN', 'MM_GLM', 'MM_NN']
    predictions = {i:template for i in methods}
    for method in methods:
        for i in range(7):
            maxHistory = min((19 * 7 - ((2*r)-7) - ((int(i) - 6) * 7)), 5 * 7)
            address = './'+str(i)+'/results/counties=1535 max_history='+str(maxHistory)+'/test/all_errors/' + method + '/all_errors_'+method+'.csv'
            temp = pd.read_csv(address)
            predictions[method]=predictions[method].append(temp)

    orig_stdout = sys.stdout
    f = open('errors.txt', 'a')
    sys.stdout = f

    for method in methods:
            y_test = np.array(predictions[method]['Target'])
            y_prediction = np.array(predictions[method]['prediction'])
            y_prediction[y_prediction < 0] = 0
            meanAbsoluteError = mean_absolute_error(y_test, y_prediction)
            print("Mean Absolute Error of ", method, ": %.2f" % meanAbsoluteError)
            sumOfAbsoluteError = sum(abs(y_test - y_prediction))
            percentageOfAbsoluteError = np.mean((abs(y_test - y_prediction)/y_test)*100)
            print("Percentage of Absolute Error of ", method, ": %.2f" % percentageOfAbsoluteError)
    sys.stdout = orig_stdout
    f.close()
      
if __name__ == "__main__":
    main()
