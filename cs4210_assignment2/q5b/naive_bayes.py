#-------------------------------------------------------------------------
# AUTHOR: Eric Pham
# FILENAME: naive_bayes.py
# SPECIFICATION: Reads weather_training.csv, trains a Naive Bayes classifier,
#                and outputs the classifications from the weather_test.csv
#                test set if the classification confidence is >= 0.75
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd

dbTraining = []
dbTest = []
X = []
Y = []

#Reading the training data using Pandas
df = pd.read_csv('q5b/weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
for row in dbTraining:
    outlook = {"Sunny": 1, "Overcast": 2, "Rain": 3}
    temperature = {"Hot": 1, "Mild": 2, "Cool":3}
    humidity = {"High": 1, "Normal": 2}
    wind = {"Strong": 1, "Weak": 2}

    X.append([
        outlook[row[1]],
        temperature[row[2]],
        humidity[row[3]],
        wind[row[4]]
    ])

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
    classification = {"Yes": 1, "No": 2}

    Y.append(classification[row[5]])

#Fitting the naive bayes to the data using smoothing
#--> add your Python code here
clf = GaussianNB()
clf.fit(X, Y)

#Reading the test data using Pandas
df = pd.read_csv('q5b/weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())

#Printing the header os the solution
#--> add your Python code here
print(f"{'Day':<6}{'Outlook':<10}{'Temperature':<12}{'Humidity':<10}{'Wind':<8}{'PlayTennis':<12}{'Confidence':<10}")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for row in dbTest:
    testData = [[
        outlook[row[1]],
        temperature[row[2]],
        humidity[row[3]],
        wind[row[4]]
    ]]

    probabilities = clf.predict_proba(testData)[0]
    predicted = clf.predict(testData)[0]

    confidence = max(probabilities)

    if confidence >= 0.75:
        if predicted == 1:
            label = "Yes"
        else:
            label = "No"

        print(f"{row[0]:<6}{row[1]:<10}{row[2]:<12}{row[3]:<10}{row[4]:<8}{label:<12}{confidence:<10.2f}")


