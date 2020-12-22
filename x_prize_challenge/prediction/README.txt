Steps to execute main.py properly:

- A list of required modules is available at requirements.txt file.
To install requirements you should just run the following command in your linux distribution:
$ pip install -r requirements.txt
If a virtual environment is necessary,
please consider the creation and activation process for it before installing required modules.

- For a properly execution of main.py, there should be a test_data.csv file in the same directory.
The main.py file first loads a KNN model from knn.model file and then use a test_data.csv file for prediction process.
the test_data.csv file must include tne following columns at minimum:
Date, C1_School closing, C2_Workplace closing, C3_Cancel public events, C4_Restrictions on gatherings',
C5_Close public transport, C6_Stay at home requirements, C7_Restrictions on internal movement,
C8_International travel controls, H1_Public information campaigns, H2_Testing policy,
H3_Contact tracing, H6_Facial Coverings
To run main.py just simply run the following command:
python main.py

- After the prediction process, two new files will be created in the same directory:
prediction.csv: containing predicted values for each record of test_data.csv file
test_data_prediction.csv: a csv file similar to test_data.csv with just one extra column, prediction,
containing the predicted value related to each record of test_data.csv file
You can use both files to examine the prediction process

- In the case to encounter with a error or failure of execution, feel free to contact me at the following email address:
arashmarioriyad@gmail.com