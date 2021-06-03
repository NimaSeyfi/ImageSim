import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import glob
import scipy
from scipy import stats

class jensen():
    def __init__(self,database,image):
        self.databasepath=database
        self.imagepath=image;
        
def calculateHist(im):
    if im is None:
        print('Could not open or find the image')
        exit(0)
        # calculate mean value from RGB channels and flatten to 1D array
    vals = im.mean(axis=2).flatten()
    # plot histogram with 255 bins
    b, bins, patches = plt.hist(vals, 255)
    #plt.xlim([0,255])
    #plt.show()
    #print(b[100])
    return b;

def jensen_cal(img1,img2):
    return jensen_val(calculateHist(img1),calculateHist(img2))

def jensen_val(p, q):
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    return (stats.entropy(p, m) + stats.entropy(q, m)) / 2    
#def jensen_val(hist1,hist2):
#    sum=0
#    for i in range(0,255):
#        pprime=hist1[i]/np.sum(hist1)
#        qprime=hist2[i]/np.sum(hist2)
#        x=pprime*np.log2(pprime)+qprime*np.log2(qprime)-(pprime+qprime)*np.log2((pprime+qprime)/2)
#        sum=sum+x
 #   return 0.5*sum
