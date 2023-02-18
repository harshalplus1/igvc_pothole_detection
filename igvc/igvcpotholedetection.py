#igvc pothole detection - white circles
import cv2 as cv
import numpy as np
# reading sample image
img=cv.imread(r"/home/harshalplus1/Downloads/igvc_pothole.jpeg")

#taking region of interest
def roi(img):
    #numpy array of region
    pts=np.array([[(75,438),(290,291),(367,291),(565,438)]],dtype=np.int32)
    #making a blank image of size img
    blank=np.zeros_like(img)
    #drawing roi of white color on blank img
    roi=cv.fillPoly(blank,pts,(255,255,255))
    #masking both
    roii=cv.bitwise_and(roi,img)
    return roii

#calling function
roii=roi(img)
#turning gray and applying blur
gray=cv.cvtColor(roii,cv.COLOR_BGR2GRAY)
blur=cv.GaussianBlur(gray,(3,3),2)
#turning the image into black and white min and max parameters
(thresh, BnW_image) = cv.threshold(blur, 200, 255, cv.THRESH_BINARY)
#canny edge detection
canny=cv.Canny(BnW_image,125,255)
#finding contour lines on canny image and drawing it on final image
contours, hierarchies=cv.findContours(canny,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
finalimg=cv.drawContours(img,contours,-1,(0,0,0),3)
cv.imshow("new",finalimg)
cv.waitKey(0)
cv.destroyAllWindows()