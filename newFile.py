import cv2
import dlib
import pickle
import os
import pandas as pd
def saveLandMarks(img, outfolderPath, fileName):
    detector = dlib.get_frontal_face_detector()

    # Load the predictor
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # read the image
    img = cv2.imread(img,0)


#Convert image into grayscale
    """    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)"""

#Use detector to find landmarks
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyWindow('img')
    faces = detector(img)
    landmarkFolder = os.path.join(outfolderPath,"landmark")
    if not os.path.exists(landmarkFolder):
        os.makedirs(landmarkFolder)
    fileNameFolderPath =   os.path.join(landmarkFolder,fileName+".csv")
    for face in faces:
        x1 = face.left() # left point
        y1 = face.top() # top point
        x2 = face.right() # right point
        y2 = face.bottom() # bottom point

        # Create landmark object
        landmarks = predictor(image=img, box=face)
      #  pickle.dump(open(img,"wb"),landmarks)
        # Loop through all the points
        xArray = []
        yArray =[]
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            xArray.append(x)
            yArray.append(y)

            # Draw a circle
            cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
        dict = {'X': xArray, 'Y': yArray}
        df = pd.DataFrame(dict)
        df.to_csv(fileNameFolderPath)
        """     cv2.imshow("img", img)
            cv2.waitKey(0)
            cv2.destroyWindow('img')"""
    # show the image
    """while(True):
        k = cv2d
        if k == -1:  # if no key was pressed, -1 is returned
            continue
        else:
            break"""