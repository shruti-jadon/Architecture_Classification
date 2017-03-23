#This code does local feature descriptor classification using Support Vector Machines. Aboned as it takes too long.

import os, re, pickle
import VQ, cv2
from collections import Counter
import numpy as np
from sklearn import neighbors
from sklearn.neighbors import DistanceMetric as DM
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
import matplotlib.pyplot as plt

codes = {'British':0,'Ancient':1,'Maratha':2,'IndoIslamic':3,'Sikh':4}

f = open('data.pkl','rb')
feature_dictionary_train  = pickle.load(f)
#print feature_dictionary_train['British'][:,-1].shape

X_train = np.ndarray((1,32))
Y_train = np.ndarray((1,1))
for key in feature_dictionary_train.keys():
    X_train = np.concatenate((X_train,feature_dictionary_train[key][:,:-1]),axis=0)
    print X_train.shape
    #print np.transpose(np.array(feature_dictionary_train[key][:,-1])).shape
    y_train = [[feature_dictionary_train[key][i,-1]] for i in range(len(feature_dictionary_train[key][:,-1]))]
    y_train = np.transpose(y_train)
    y_train = np.transpose(y_train)
    #print y_train.shape
    Y_train = np.concatenate((Y_train,y_train),axis=0)
    print Y_train.shape

for c in np.arange(10,11,0.5):
    CSVC_model = SVC(C=c)

    filename = 'svc_clf_%f.pkl' % (c)
    svc_clf = open(filename,'wb')

    CSVC_model.fit(X_train[1:,:], np.ravel(Y_train[1:,:]))

    pickle.dump(CSVC_model,svc_clf)
    print 'model %f done' % (c)

'''
name = 'svc_clf_0.500000.pkl'
f = open(name,'rb')
CSVC_model = pickle.load(f)

directory = '../Test Images'
archtypes = os.listdir(directory)

final_score = 0

for name in archtypes:
    csvfiledirec = os.path.join(directory,name)
    filelist = os.listdir(csvfiledirec)

    print 'Folder: ', name
    score = 0   
    #f = open(name + '_29.txt','w') 
    for csvfile in filelist:
        #print csvfile
        filepath = os.path.join(csvfiledirec,csvfile)
        #print filepath
        image = cv2.imread(filepath,0)
    #print filepath
        feature_matrix = VQ.final_features(image,500)    
        X_test = feature_matrix
        #print X_test
        Predictions = CSVC_model.predict(X_test)

        label = Counter(Predictions)
        print 'class of the test image is: ', label.most_common()
            #f.write(str(label.most_common()))
            #f.write('\n')

        if label.most_common(1)[0][0] == codes[name]:
            score += 1
    f.close()
    print 'accuracy score for class %s = %f' % (name, float(score) / 10)
    final_score += score
print 'final accuracy score: ', float(final_score) / 50
'''    
