import numpy as np
import cv2

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#name format
name = "ID00_{}.bmp"
i = 0

#start videoCapture
cap = cv2.VideoCapture(0)
done = False

while not done:
    #new frame
    ret, img = cap.read()

    #find faces on grayscale frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #if only one face is detected
    #could be changed with a loop to capture all faces
    # with "for (x,y,w,h) in faces:"
    if len(faces) == 1:
        (x,y,w,h) = faces[0]

        #mask skin by color parameters
        im = cv2.cvtColor(img[y:y+h, x:x+w],cv2.COLOR_BGR2RGB)
        R,G,B = [im[:,:, channel] for channel in range(3)]

        ims1 = (R>95) & (G>40) & (B>20) & (R > B) & (R > G)
        ims2 = im.ptp(axis=-1) > 15
        ims3 = np.abs(np.int8(R) - np.int8(G)) > 15
        ims = ims1 & ims2 & ims3
        color_mask = (np.zeros(ims.shape,dtype='uint8')+ims)*255

        #find contours and get hull of them all
        contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(np.vstack(contours))

        #draw convex hull on the new black picture
        final_mask = np.zeros((color_mask.shape[0], color_mask.shape[1],3), np.uint8)
        cv2.drawContours(final_mask, [hull], -1, (255,255,255), thickness=cv2.FILLED)

        #mask gray img for final output
        out = cv2.bitwise_and(gray[y:y+h, x:x+w],final_mask[:,:,0])
        cv2.imshow("out",out)

        #waits for input indefinetly, change "0" to an int 
        #to wait for miliseconds => video
        #"s" key saves photo in the right format
        #"q" quits the program
        #any other key takes a new picture
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'):
            i+=1
            cv2.imwrite(name.format(str.rjust(str(i),3,'0')),cv2.resize(out,(100,100)))
        elif key == ord('q'):
            done = True