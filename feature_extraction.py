import ORB_Practice
import VQ
import os, numpy
import cv2

path = './Test Images'

files = os.listdir(path)

for f in files:
  if not (f.startswith('.')):
    folder = os.path.join(path,f)
    print folder
    filenames = os.listdir(folder)
    
    #N = input('Decide the number of rows of features (number of feature centroids): ')    
    N = 15
    print filenames[0]
    filepath = os.path.join(folder, filenames[0]) 
    image = cv2.imread(filepath,0)
    print filepath
    feature_matrix = VQ.final_features(image,N)
    print feature_matrix.shape
    print ('\nimage %s done\n' % filenames[0])

    for filename in filenames[1:]:
      if not (filename.startswith('.')):
        filepath = os.path.join(folder, filename)
        image = cv2.imread(filepath,0)
    
        features = VQ.final_features(image,N)
        print features.shape
        print ('\nimage %s done\n' % filename)

        feature_matrix = numpy.vstack((feature_matrix,features))

        print feature_matrix.shape
    
    name = './Test_CSV/' + f + '.csv'
    
    numpy.savetxt(name, feature_matrix, fmt='%.5f', delimiter=' ')
