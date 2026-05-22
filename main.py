import cv2
import numpy as np 

img = cv2.imread("blob.jpg",0)
# Initialize filtering function
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100
params.filterByCircularity = True 
params.minCircularity = 0.9
params.filterByConvexity = True 
params.minConvexity = 0.2
# Inertia filtering parameters
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect the blobs 
keyPoints = detector.detect(img)
# Draw the blobs on the image

#Draw the lines around the blobs
blank = np.zeros((1,1))
blobs_img = cv2.drawKeypoints(img,keyPoints,blank,(255,223,73),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
number_blobs = len(keyPoints)
text = f"Total Number Of Blobs Detected: {number_blobs}"
cv2.putText(blobs_img,text,(20,550),cv2.FONT_HERSHEY_SIMPLEX,0.8,(57,23,100),2)
cv2.imshow("Blob Detector",blobs_img)
cv2.waitKey(0)
cv2.destroyAllWindows()