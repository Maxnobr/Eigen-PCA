import numpy as np
import cv2

load_weight = np.load('wt_A.npy')
load_mean_vect = np.load('mean_vect.npy')

out = cv2.resize(np.uint8(load_mean_vect.reshape((50,50))),(500,500))
cv2.imshow("out",out)

cv2.waitKey(0)
print("done")