import numpy as np
import os, sys, csv
import save_to_csvfile
from collections import defaultdict
from random import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import log_loss
import Plotting

direc = './CSV_FILES'

list_csv = os.listdir(direc)

features_dict = defaultdict(float)
actual_labels = defaultdict(int)

feature_array = np.ndarray((500,481))
l = 0
#value_array = np.ndarray((489,1), dtype = int)
count = -1

# Append the lists together first, then convert to float and append 15-15 lines together. Once 15 are appended, add all these to the array feature_array.

for name in list_csv:
    
    feature_list = ''
    k = 0
    
    csv_name = os.path.join(direc, name)
    f = open(csv_name, 'r')
    features_dict[name] = f.readlines()
    rows = len(features_dict[name]) / 15
    actual_labels[name] = rows
    col = 15 * 32
    
    if not (name.startswith('.')):
       count +=1
       for line in features_dict[name]:
           k += 1
           feature_list += line
           if k == 15:
               numbers = feature_list.split()
               numbers = [float(i) for i in numbers]
               feature_array[l][0:480] = numbers
               feature_array[l][480] = count
               l += 1
               k = 0
               feature_list = ''

feature_train = np.concatenate((feature_array[:80,:], feature_array[100:180,:], feature_array[200:280,:], feature_array[300:380,:] , feature_array[400:480,:]),axis=0)

feature_test = np.concatenate((feature_array[80:100,:], feature_array[180:200,:], feature_array[280:300,:], feature_array[380:400,:] , feature_array[480:500,:]),axis=0)

#np.random.shuffle(feature_train)

C_score = []

X_train = feature_train[:,:480]

Y_train = feature_train[:,480]

X_test = feature_test[:,:480]

Y_test = feature_test[:,480]

grid = np.arange(10,50,5)
for K in grid: 

    clf = RandomForestClassifier(n_estimators=K)

    #Calculate the mean scores for each value of hyperparameter C
    scores = cross_val_score(clf,X_train,Y_train,cv=5)
    print scores.mean()
    C_score.append(scores.mean())
 
#Display the maximum score achieved at which hyperparameter value
print " max score is " , max(C_score) , " at C = " , grid[C_score.index(max(C_score))]
'''
Plotting.line_graph(C_score,"Random_Forest",10,200,5)
'''

clf = RandomForestClassifier(n_estimators=grid[C_score.index(max(C_score))])
clf.fit(X_train, Y_train)
labels_test = clf.predict(X_test)

print "Accuracy score is ", accuracy_score(Y_test, labels_test)
