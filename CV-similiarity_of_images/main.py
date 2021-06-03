import numpy as np
import cv2
from PIL import Image
import ssim.ssimlib as pyssim
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.measure import compare_ssim as ssim
from scipy.spatial.distance import euclidean
from skimage import feature
import jensen
#SIFT_RATIO = 0.7
eps=0.0000001
def sigmoid(z):
    return 1 / (1 + np.around(np.exp(-z),decimals=4))

def resizeImage(image):
    (h, w) = image.shape[:2]

    width = 360  #  This "width" is the width of the resize`ed image
    # calculate the ratio of the width and construct the
    # dimensions
    ratio = width / float(w)
    dim = (width, int(h * ratio))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    #resized = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
    return resized
'''
def siftCal(imageA,imageB):
    imageA=resizeImage(imageA)
    imageB=resizeImage(imageB)
    similarity = 0.0
    # Using OpenCV for feature detection and matching
    sift = cv2.xfeatures2d.SIFT_create()
    k1, d1 = sift.detectAndCompute(imageA, None)
    k2, d2 = sift.detectAndCompute(imageB, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d1, d2, k=2)

    for m, n in matches:
        if m.distance < SIFT_RATIO * n.distance:
            similarity += 1.0

    # Custom normalization for better variance in the similarity matrix
    if similarity == len(matches):
        similarity = 1.0
    elif similarity > 1.0:
        similarity = 1.0 - 1.0/similarity
    elif similarity == 1.0:
        similarity = 0.1
    else:
        similarity = 0.0 
    return similarity   
'''
def cw_ssim(imageA,imageB):
    
    return pyssim.SSIM(Image.fromarray(imageA)).cw_ssim_value(Image.fromarray(imageB))
def psnr(imageA, imageB):
    return (10*np.log10(255**2/(mse(imageA, imageB)+eps)))
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err
def LBP(imageA,imageB):
    lbpA=LBP_feature(imageA).ravel()
    lbpB=LBP_feature(imageB).ravel()
    return euclidean(lbpA, lbpB)*100
def LBP_feature(image):
    image = resizeImage(image)
    (h, w) = image.shape[:2]
    #cellSize = 16 * 2
    cellSize = h/10
    # 3 convert the image to grayscale and show it
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Image", gray)    
    lbp = feature.local_binary_pattern(gray, 10, 5, method="default") # method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, 10),range=(0,255))
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    # return the histogram of Local Binary Patterns
    return hist
    '''
    cv2.imshow("LBP", features.astype("uint8"))
    (fig, ax) = plt.subplots()
    fig.suptitle("Local Binary Patterns")
    plt.ylabel("% of Pixels")
    plt.xlabel("LBP pixel bucket")
    ax.hist(features.ravel(), normed=True, bins=20, range=(0, 256))
    ax.set_xlim([0, 256])
    ax.set_ylim([0, 0.030])
    # save figure
    fig.savefig('lbp_histogram.png')   # save the figure to file
    plt.show() 
    '''

#img1_address=input("Enter address of first Image:\n")
#img1=cv2.imread(img1_address,1)
#img2_address=input("Enter address of second Image:\n")
#img2=cv2.imread(img2_address,1)
img1=cv2.imread('C:/Users/msi/Desktop/CV Proj/main.jpg',cv2.IMREAD_UNCHANGED)
img2=cv2.imread('C:/Users/msi/Desktop/CV Proj/p&s5percent.jpg',cv2.IMREAD_UNCHANGED)
algo=input("Select Algorithm:")
img1 = resizeImage(img1)
img2 = resizeImage(img2)
if(algo.lower()=="all"):
    print("--FR-IQA--")
    mse_val=mse(img1,img2)
    print("MSE:",mse_val)

    psnr_val=psnr(img1,img2)
    print("PSNR:",psnr_val)    
    ssim_val=ssim(img1,img2,multichannel=True)
    print("SSIM:",ssim_val) 
    cwssim_val=cw_ssim(img1,img2)
    print("CW-SSIM:",cwssim_val)   
    lbp_val=LBP(img1,img2)
    print("LBP:",lbp_val)
    jensen_value=jensen.jensen_cal(img1, img2)
    print("JENSEN:",1.0-jensen_value)
elif(algo.lower()=="mse"):
    mse_val=mse(img1,img2)
    print("MSE:",mse_val)
elif(algo.lower()=="ssim"):
    ssim_val=ssim(img1,img2,multichannel=True)
    print("SSIM:",ssim_val)
elif(algo.lower()=="lbp"):
    lbp_val=LBP(img1,img2)
    print("LBP:",lbp_val)
elif(algo.lower()=="psnr"):
    psnr_val=psnr(img1,img2)
    print("PSNR:",psnr_val)
elif(algo.lower()=="cwssim"):
    cwssim_val=cw_ssim(img1,img2)
    print("CW-SSIM:",cwssim_val)
elif(algo.lower()=="jensen"):
    jensen_value=jensen.jensen_cal(img1, img2)
    print("JENSEN:",1.0-jensen_value)
else:
    print("\n------Algorithm not detected--------\n")
#cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#cv2.imshow('image',img1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
print("\n----SHOW IMAGES----")
gs = gridspec.GridSpec(1, 2)
plt.axis("off")
plt.subplot(gs[0,0]),plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)),plt.title('Image 1')
plt.xticks([]), plt.yticks([])
plt.subplot(gs[0,1]),plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)),plt.title('Image 2')
plt.xticks([]), plt.yticks([])

#mng = plt.get_current_fig_manager()
#mng.full_screen_toggle()
plt.show()

