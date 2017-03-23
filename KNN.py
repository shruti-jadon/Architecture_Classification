#This code proposes a new method of image classification using KNN based on local feature descriptors. Each local 
#feature for a particular label is considered a separate input with that label. For a single image with 500 descriptors, 
#we have 500 inputs instead of just one. For one class, we have 500 x 100 = 50000 inputs and for all the  classes, a total 
#of 5 x 50000 = 250000 inputs, each with a separate label. For classification, each descriptor of the test image is put through 
#the KNN setting and given a label. Final label of the image is taken as the mode of labels that occur for its descriptors.

import os, re
import VQ, cv2
from collections import Counter
import numpy as np
import pickle
from sklearn.externals import joblib
from sklearn import neighbors
from sklearn.neighbors import DistanceMetric as DM
from sklearn.neighbors import KNeighborsClassifier as KNN


codes = {'British':0,'Ancient':1,'Maratha':2,'IndoIslamic':3,'Sikh':4}

'''
feature_dictionary_train = {}
feature_dictionary_test = {}

#Load training data
directory = './train'

file_list = os.listdir(directory)  

for csvfile in file_list:
  if not (csvfile.startswith('.')):
    print file_list
    feature_array_train = np.ndarray((1,33))
    
    archtype = re.findall('(.*)_500x32_train\.csv',csvfile)
    
    file_path = os.path.join(directory,csvfile)
    f = open(file_path)
    csv_data = f.readlines()
    
    for line in csv_data:
        numbers = line.split()
        numbers = [float(i) for i in numbers]
        numbers.append(codes[archtype[0]])
        feature_array_train = np.vstack((feature_array_train,numbers))
    print feature_array_train[:,-1]
    feature_dictionary_train[archtype[0]] = feature_array_train[1:,:]
    print 'Training set size: ', feature_array_train[1:,:].shape
    
    
#Load testing data
directory = './Test Images'

file_list = os.listdir(directory)  

for csvfile in file_list[0]:
    #feature_array_test = np.ndarray((1,32))
    
    #archtype = re.findall('(.*)_500x32_test\.csv',csvfile)
    #print archtype

    file_path = os.path.join(directory,csvfile)
    image = cv2.imread(file_path,0)
    #print filepath
    feature_matrix = VQ.final_features(image,500)
    #print feature_matrix.shape
    #print ('\nimage %s done\n' % filenames[0])
    
    #f = open(file_path)
    #csv_data = f.readlines()
    
    #for line in csv_data:
    #    numbers = line.split()
    #    numbers = [float(i) for i in numbers]
    #    feature_array_test = np.vstack((feature_array_test,numbers))
    #feature_dictionary_test[archtype[0]] = feature_array_test[1:,:]
    #print 'Testing set size: ', feature_array_test[1:,:].shape
    #feature_array_test = feature_array_test[1:,:]
print 'Done'


f = open('data.pkl','wb')
pickle.dump(feature_dictionary_train, f)


f = open('data.pkl','rb')
feature_dictionary_train  = pickle.load(f)

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

KNN_model = KNN(43)
print X_train.shape

knn_clf = open('knn_clf_43.pkl','wb')

KNN_model.fit(X_train[1:,:], np.ravel(Y_train[1:,:]))

pickle.dump(KNN_model,knn_clf)

print "done"
'''
f = open('knn_clf_43.pkl','rb')
KNN_model = pickle.load(f)
f.close()

directory = './Test Images'
archtypes = os.listdir(directory)

final_score = 0
text_file = open("./outputs/Output_43.txt", "w")

for name in archtypes:
    csvfiledirec = os.path.join(directory,name)
    filelist = os.listdir(csvfiledirec)

    print 'Folder: ', name
    score = 0    
    for csvfile in filelist:
        filepath = os.path.join(csvfiledirec,csvfile)
        
        image = cv2.imread(filepath,0)
        
        feature_matrix = VQ.final_features(image,500)    
        X_test = feature_matrix
        
        Predictions = KNN_model.predict(X_test)

        label = Counter(Predictions)

        print 'class of the test image is: ', label.most_common(1)
        
        if label.most_common(1)[0][0] == codes[name]:
            score += 1

    print 'accuracy score for class %s = %f' %(name, float(score) / 10)
    
    text_file.write("\naccuracy score for class %s = %f" % (name, float(score) / 10))
    final_score += score
print 'final accuracy score: %f ' %(float(final_score) / 50)

text_file.write("\nfinal accuracy score: %f" %(float(final_score) / 50))
text_file.close()