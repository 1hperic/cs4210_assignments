#-------------------------------------------------------------------------
# AUTHOR: Eric Pham
# FILENAME: decision_tree.py
# SPECIFICATION: produces a depth-2 decision tree; 
#                output based on data from the file contact_lens.csv
# FOR: CS 4210- Assignment #1
# TIME SPENT: 5 hours total, 2 hours on 7b
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#encode the original categorical training features into numbers and add to the 4D array X.
#--> add your Python code here
# X =
age = {"Young": 0, "Prepresbyopic": 1, "Presbyopic": 2}
spectacle = {"Myope": 0, "Hypermetrope": 1}
astigmatism = {"Yes": 0, "No": 1}
tear = {"Normal": 0, "Reduced": 1}

for row in db:
  features = [age[row[0]], spectacle[row[1]], astigmatism[row[2]], tear[row[3]]]
  X.append(features)

#encode the original categorical training classes into numbers and add to the vector Y.
#--> add your Python code here
# Y =
recommend = {"Yes": 0, "No": 1}

for row in db:
   Y.append(recommend[row[4]])

#fitting the depth-2 decision tree to the data using entropy as your impurity measure
#--> add your Python code here
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=2)
clf = clf.fit(X, Y)

#plotting decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()