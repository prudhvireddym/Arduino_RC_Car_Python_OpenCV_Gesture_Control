import numpy as np
import os
from keras_preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle


############################################################

path = 'dataset'
testRatio = 0.25
valRatio = 0.25
imageDimensions= (32,32,3)
batchSizeVal= 50
epochsVal = 10
stepsPerEpochVal = 2000
############################################################
images = []
classNo = []

myList = os.listdir(path)
print("No of classes detected: ",len(myList))
noOfClasses = len(myList)

############### appending all the images in the directories to list images ###########################
print("Importing ",len(myList)," Classes")
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
x_train,x_test,y_train,y_test = train_test_split(images,classNo,test_size=testRatio)#train_test_split can also be used to shuffle the images so it doesnot take all  tthe irst 80% images
x_train,x_validation,y_train,y_validation = train_test_split(x_train,y_train,test_size=testRatio)
print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)

noOfSamples = []
for x in range(0,noOfClasses):
    noOfSamples.append(len(np.where(y_train==x)[0]))#appending the image class into a list
print("The no of images in each class: ",noOfSamples) #Printing the no of samples of each class



############# Preprocessing the images #####################################################
def preprocessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Convert image to grey - sets the depth to 0
    img = cv2.equalizeHist(img) #Equalize the light in the image
    img = img/255
    return img
    
x_train = np.array(list(map(preprocessing,x_train)))#Map function is used to send the list of elements to a function
x_test = np.array(list(map(preprocessing,x_test)))
x_validation = np.array(list(map(preprocessing,x_validation)))


#Adding depth of 1 for the CNN to run properly, depth was set to none while converting to greyscale
print(x_train.shape)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_validation = x_validation.reshape(x_validation.shape[0],x_validation.shape[1],x_validation.shape[2],1)

print(x_train.shape)


#Argumenting the images like zooming,rotating, etc to make the dataset more generic
dataGen = ImageDataGenerator(width_shift_range=0.1
                             ,height_shift_range=0.1
                             ,zoom_range=0.2
                             ,shear_range=0.1
                             ,rotation_range=10)

#help our generator perform some statistics before calculation
dataGen.fit(x_train)

#One Hot encoding our data
y_train = to_categorical(y_train,noOfClasses)
y_test = to_categorical(y_test,noOfClasses)
y_validation = to_categorical(y_validation,noOfClasses)


#Creating our model 
#### CREATING THE MODEL 
def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2,2)
    noOfNodes= 500

    model = Sequential()
    model.add((Conv2D(noOfFilters,sizeOfFilter1,input_shape=(imageDimensions[0],
                      imageDimensions[1],1),activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNodes,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))

    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model = myModel()
print(model.summary())




#### STARTING THE TRAINING PROCESS
batchSizeVal = 50
epochsVal = 10
stepsPerEpochVal = 2000

history = model.fit_generator(dataGen.flow(x_train,y_train,
                                 batch_size=batchSizeVal),
                                 steps_per_epoch=stepsPerEpochVal,
                                 epochs=epochsVal,
                                 validation_data=(x_validation,y_validation),
                                 shuffle=1)


#### EVALUATE USING TEST IMAGES
score = model.evaluate(x_test,y_test,verbose=0)
print('Test Score = ',score[0])
print('Test Accuracy =', score[1])

#### SAVE THE TRAINED MODEL 
pickle_out= open("model_trained_rps.p", "wb")
pickle.dump(model,pickle_out)
pickle_out.close()

