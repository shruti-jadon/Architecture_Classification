import cv2
import drawMatches as dm
import numpy
import matplotlib.pyplot as plt

def orb_descriptor(img):
    
    orb = cv2.ORB()

    kp, des = orb.detectAndCompute(img,None)
    
    #print des
    return(kp, des)

img = cv2.imread('../Images/IndoIslamic/80.jpg')

kp, des = orb_descriptor(img)
img2 = cv2.drawKeypoints(img,kp,color=(255,0,0), flags=0)
plt.imshow(img2),plt.show()

#print len(kp1) , len(kp2) , len(kp3)
#print des1.shape , des2.shape , des3.shape

#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#matche1 = bf.match(des1,des5)
#matche2 = bf.match(des1,des4)
#matche3 = bf.match(des3,des4)

#matche1 = sorted(matche1, key = lambda x:x.distance)
#matche2 = sorted(matche2, key = lambda x:x.distance)
#matche3 = sorted(matche3, key = lambda x:x.distance) 

#print 'matches between image 1 and image 2: \n', len(matche1)
#print 'matches between image 1 and image 3: \n', len(matche2)
#print 'matches between image 3 and image 4: \n', len(matche3)

#newimg1 = dm.drawMatches(img1,kp1,img5,kp2,matche1)
#newimg2 = dm.drawMatches(img1,kp1,img3,kp3,matche2)
#newimg3 = dm.drawMatches(img3,kp3,img4,kp4,matche3)
 
#plt.imshow(newimg1),plt.show()
#plt.imshow(newimg2),plt.show()
#plt.imshow(newimg3),plt.show()
