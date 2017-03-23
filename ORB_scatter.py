#This code prepares a scatter plot for the ORB features, taking the first two dimensions into account.

import matplotlib.pyplot as plt
import os,re
import numpy as np
import copy

directory = '../New'

list_csv = os.listdir(directory)
X_Y_array = np.ndarray((1,2))
name_list = []

for name in list_csv:
    name_list.append(re.findall('(.*)_1x32\.csv',name))
    temp_array = np.ndarray((100,2))
    i = 0
    filepath = os.path.join(directory,name)
    f = open(filepath)
    f_lines = f.readlines()
    for line in f_lines:
        k = 0
        new_line = line.split()
        for num in new_line[0::31]:
            temp_array[i][k] = float(num)
            print temp_array[i][k]
            k += 1
        i += 1
    
    #print X_Y_array
    #print temp_array
    X_Y_array = np.append(X_Y_array,temp_array,axis=0)
    #print X_Y_array
print X_Y_array[1:,:].shape

X = list(X_Y_array[1:,0])
Y = list(X_Y_array[1:,1])
X_new = copy.deepcopy(X)
Y_new = copy.deepcopy(Y)
print len(X)
for x in range(len(X)):
    #print x
    if X[x] < 1.0: 
        X_new.pop(x)
        print x
    if Y[x] < 0.5:
        Y_new.pop(x)
     
scatterplot = plt.subplot()

scatterplot.scatter(X[0:100],Y[0:100],color='red')
#scatterplot.scatter(X_new[0:100],Y[0:100],color='blue')
scatterplot.scatter(X_new[100:200],Y[100:200],color='blue')
scatterplot.scatter(X[200:300],Y[200:300],color='green')
scatterplot.scatter(X[300:400],Y[300:400],color='cyan')
scatterplot.scatter(X[400:500],Y[400:500],color='yellow')
#print name_list[0]
#scatterplot.title('red = %s, blue = %s, green = %s, cyan = %s, yellow = %s' % (name_list[0],name_list[1],name_list[2],name_list[3],name_list[4]))
plt.legend(name_list)
plt.show()
