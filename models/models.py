import pickle
import os, numpy
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

class model:
    def __init__(self):
        print ("initiated")

    def preprocess_data(self,path):
        files = os.listdir(path)
        orb = cv2.ORB_create()
        #sift = cv2.xfeatures2d.SIFT_create()
        X=numpy.zeros(shape=(1,500*32))
        Y=numpy.zeros(shape=(1,))
        count=-1
        for f in files:
             if not (f.startswith('.')):
                folder = os.path.join(path, f)
                filenames = os.listdir(folder)
                count+=1
                for fi in filenames:
                    if not (fi.startswith('.')):
                        filepath = os.path.join(folder, fi)
                        #print filepath
                        image = cv2.imread(filepath, 0)
                        kp, features = orb.detectAndCompute(image,None)
                        #print features.shape
                        features=features.reshape(1,500*32)
                        #print features.shape
                        X = numpy.vstack((X, features))
                        Y=numpy.vstack((Y,numpy.asarray([count])))
        #print X.shape
        Y=Y.reshape(Y.shape[0],)
        return X,Y

    def model_svm(self,X,y):
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        C_score = []
        grid = np.arange(0.01, 3, 0.1)
        for K in grid:
            clf = SVC(C=K)
            # Calculate the mean scores for each value of hyperparameter C
            scores = cross_val_score(clf, X_train, Y_train, cv=5)
            print scores.mean()
            C_score.append(scores.mean())

        # Display the maximum score achieved at which hyperparameter value
        print " max score is ", max(C_score), " at C = ", grid[C_score.index(max(C_score))]
        clf = SVC(C=grid[C_score.index(max(C_score))])
        clf.fit(X_train, Y_train)
        y_pred = clf.predict(X_test)
        print accuracy_score(Y_test, y_pred)

    def model_KNN(self,X,Y):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
        clf = neighbors.KNeighborsClassifier(10)
        clf.fit(X_train, Y_train)
        y_pred = clf.predict(X_test)
        print accuracy_score(Y_test, y_pred)
    def model_randomforest(self,X,Y):
        print X.shape

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
        K=1000
        clf = RandomForestClassifier(n_estimators=K)
        # Calculate the mean scores for each value of hyperparameter C
        #scores = cross_val_score(clf, X, Y, cv=5)
        #print scores.mean()
        clf.fit(X_train, Y_train)
        y_pred = clf.predict(X_test)
        print accuracy_score(Y_test, y_pred)




m=model()
X,Y=m.preprocess_data("/Users/shrutijadon/Desktop/github/Architecture_Classification/Data")
#m.model_svm(X,Y)
#m.model_KNN(X,Y)
m.model_randomforest(X,Y)