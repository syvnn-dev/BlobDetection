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