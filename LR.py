import numpy as np
import os, sys, csv
from collections import defaultdict
from random import shuffle
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.kernel_approximation import RBFSampler,Nystroem

direc = './CSV_FILES'

list_csv = os.listdir(direc)

features_dict = defaultdict(float)
actual_labels = defaultdict(int)

feature_array = np.ndarray((500,481))
l = 0
#value_array = np.ndarray((489,1), dtype = int)
count = -1
C_score = []
acc = []
names = []
avg_prob = np.ndarray((1,5))
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
       names.append(name)
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

np.random.shuffle(feature_train)


X_train = feature_train[:,:480]

Y_train = feature_train[:,480]

X_test = feature_test[:,:480]

Y_test = feature_test[:,480]

clf = linear_model.LogisticRegression(solver = 'newton-cg',multi_class='multinomial')
clf.fit(X_train, Y_train)

labels_test = clf.predict_proba(X_test)

grid = np.arange(0,100,20)
for k in grid:

   print "\ntest case: ", str(k), ": ", labels_test[k][0],labels_test[k][1],labels_test[k][2],labels_test[k][3],labels_test[k][4]
   
   avg_prob = np.vstack((avg_prob,np.mean(labels_test[k:k+20,:],axis=0)))
   
print "csv files read are :" , names
print avg_prob


