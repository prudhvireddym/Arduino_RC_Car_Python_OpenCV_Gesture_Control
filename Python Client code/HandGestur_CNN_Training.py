import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
############################################################

path = 'data'
testRatio = 0.2
############################################################
images = []
classNo = []

myList = os.listdir(path)
print("Total No of classes detected: ",len(myList))
noOfClasses = len(myList)

############### appending all the images in the 10 directories to list images ###########################
print("Importing classes...")
for x in range (0,noOfClasses):                         #Iterating through the directory
    myPicList = os.listdir(path+"/"+str(x))
    for y in myPicList:                                 #iterating through the images
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)
        curImg = cv2.resize(curImg,(32,32))             #resizing the images for efficiency
        images.append(curImg)                           #appending the images to the images list
        classNo.append(x)                               #appending the respective directory numbers to the images
    print(x,end=" ")                                    #Printing in a single line
print("")


#Converting all the images to a numpy array
images = np.array(images)
classNo = np.array(classNo)

print(images.shape) #2000 images of 32X32 size and 3 channels that is coloured images
print(classNo.shape) #2000 values


################### Splitting the data into train test split #############################
X_train,X_test,y_train,y_test = train_test_split(images,classNo,test_size=testRatio)


