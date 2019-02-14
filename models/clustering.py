from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np
import os, sys, csv
from collections import defaultdict

direc = '../CSV files'

list_csv = os.listdir(direc)

features_dict = defaultdict(float)
actual_labels = defaultdict(int)

feature_array = np.ndarray((500,480))
X_labels = np.ndarray(500)
X_labels.astype('int')

#print X_labels
l = 0

# Append the lists together first, then convert to float and append 15-15 lines together. Once 15 are appended, add all these to the array feature_array.
n = 0
for name in list_csv:
    
    feature_list = ''
    k = 0
    
    csv_name = os.path.join(direc, name)
    f = open(csv_name, 'r')
    features_dict[name] = f.readlines()
    rows = len(features_dict[name]) / 15
    actual_labels[name] = rows
    col = 15 * 32
        
    for line in features_dict[name]:
        k += 1
        feature_list += line
        if k == 15:
            numbers = feature_list.split()
            numbers = [float(i) for i in numbers]
            feature_array[l] = numbers
            X_labels[l] = n
            l += 1
            k = 0
            feature_list = ''
            
    n += 1
#print X_labels.shape 
X = feature_array

model = KMeans(n_clusters = 5, init = 'random')
model.fit(X)

pred_labels = model.predict(X)
pred_labels_dict = defaultdict(int)

for l in pred_labels:
    pred_labels_dict[l] += 1

print 'actual labels: ', actual_labels
print 'predicted labels: ', pred_labels_dict

score = adjusted_rand_score(X_labels, pred_labels)

print score
