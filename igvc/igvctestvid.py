#igvc pothole detection - white circles
import cv2 as cv
import numpy as np
# reading sample image
vid=cv.VideoCapture(r"/home/harshalplus1/Downloads/vid.mp4")

#taking region of interest
def roi(img):
    #numpy array of region
    pts=np.array([[(175,900),(400,200),(1100,200),(1400,900)]],dtype=np.int32)
    #making a blank image of size img
    blank=np.zeros_like(img)
    #drawing roi of white color on blank img
    roi=cv.fillPoly(blank,pts,(255,255,255))
    #masking both
    roii=cv.bitwise_and(roi,img)
    return roii

def birdview(img):
    w=1500
    h=1000
    pt1=np.float32([[(400,200),(1100,200),(175,900),(1400,900)]])
    pt2=np.float32([[0,0],[1500,0],[300,980],[1200,980]])
    matrix=cv.getPerspectiveTransform(pt1,pt2)
    output=cv.warpPerspective(img,matrix,(w,h))
    return output


while True:
    ret,frame=vid.read()
    #calling functions
    roii=roi(frame)
    img1=birdview(roii)
    #turning gray and applying blur
    gray=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(gray,19)
    #turning the image into black and white min and max parameters
    (thresh, BnW_image) = cv.threshold(blur, 200, 255, cv.THRESH_BINARY)
    #canny edge detection
    canny=cv.Canny(BnW_image,125,255)
    #finding contour lines on canny image and drawing it on final image
    contours, hierarchies=cv.findContours(canny,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    finalimg=cv.drawContours(img1,contours,-1,(0,0,0),6)
    cv.imshow("new",img1)
    if cv.waitKey(1) and 0xFF==ord("s"):
        break
vid.release()
cv.destroyAllWindows()