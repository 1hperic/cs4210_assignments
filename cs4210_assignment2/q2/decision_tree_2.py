#-------------------------------------------------------------------------
# AUTHOR: Eric Pham
# FILENAME: decision_tree_2.py
# SPECIFICATION: Program reads contact_lens_training_1.csv, contact_lens_
#                training_2.csv, and contact_lens_training_3.csv and trains, tests, and 
#                outputs the performance of 3 models using each training set on the test 
#                set contact_lens_test.csv
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn import tree
import pandas as pd

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

age = {"Young": 1, "Prepresbyopic": 2, "Presbyopic": 3}
spectacle = {"Myope": 1, "Hypermetrope": 2}
astigmatism = {"Yes": 1, "No": 2}
tear = {"Normal": 1, "Reduced": 2}
classification = {"Yes": 1, "No": 2}

#Reading the test data in a csv file using pandas
dbTest = []
df_test = pd.read_csv('contact_lens_test.csv')
for _, row in df_test.iterrows():
    dbTest.append(row.tolist())

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file using pandas
    # --> add your Python code here
    dbTraining = pd.read_csv(ds).values.tolist()

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here

    for data in dbTraining:
        X.append([
            age[data[0]],
            spectacle[data[1]],
            astigmatism[data[2]],
            tear[data[3]]
        ])
        Y.append(classification[data[4]])

    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here

    #Loop your training and test tasks 10 times here
    for i in range (10):

       # fitting the decision tree to the data using entropy as your impurity measure and maximum depth = 5
       # --> add your Python code here
       clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
       clf = clf.fit(X, Y)

       correct = 0

       accuracies = []

       #Read the test data and add this data to dbTest
       #--> add your Python code here
       for data in dbTest:
           #Transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here
            test_instance = [[
                age[data[0]],
                spectacle[data[1]],
                astigmatism[data[2]],
                tear[data[3]]
            ]]

           #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           #--> add your Python code here
            predictedClass = clf.predict(test_instance)[0]
            actualLabel = classification[data[4]]

            if predictedClass == actualLabel:
                correct += 1

            accuracy = correct / len(dbTest)
            accuracies.append(accuracy)

    #Find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
    avgAccuracy = sum(accuracies) / len(accuracies)

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print(f"final accuracy when training on {ds}: {avgAccuracy}")




