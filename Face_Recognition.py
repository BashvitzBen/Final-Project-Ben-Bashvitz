# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 21:34:03 2021

@author: bashv
"""

import os # a library that is used to create paths to access or create files.
from class_model import model # a module that contains the 'model' class which takes care of all functions that relate to the model.
from class_data import Data # a module that contains the 'Data' class which takes care of all functions that relate to creating,
#                             processing and saving the data for the models.


def main():
    #the main function of the project
    #input: takes a number of inputs from the user.
    #output: uses the inputs to guide the user into training the model and testing it.
    FT_Training_Dir = os.path.join("FaceDetection","train")
    FT_Categories = ["Face", "Not"]
    Img_Size = 128
    ft_data = Data()
    ft_data.create_training_data(FT_Training_Dir, FT_Categories, Img_Size, 'FaceX.pickle', 'FaceY.pickle')
    if len(ft_data.X) != 0:
        ft_model = model()
        ft_model.initialize_face_detection_model(ft_data.X, ft_data.y)
        print("If you want to train a new face recognition model, enter '1'.")
        print("If you want to test a model that is already trained, enter '0'.")
        flag_new = int(input())
        print("Enter the name of the person whose face you want the model to recognise")
        name = input()
        try:
            os.mkdir(os.path.join(os.path.join("FaceRecognition", "train"), name)) #creates new folder called 'name' inside the 'FaceRecognition/train/' directory.
                                                                                   #this folder wil contain images of the corresponding person's face.
            print("A folder called ", name, "has been created in the 'train' folder inside the 'FaceRecognition' folder you have added earlier.")
        except Exception as e:
            print("A folder called ", name, "already exists. Either a model for this person is ready. Or, you need to train one.")
            pass
        name_x = name + "X.pickle"
        name_y = name + "Y.pickle"
        FR_Training_Dir = os.path.join("train", name)
        FR_Categories = [name, "Not"]
        if flag_new:
            print("In order for the face recognition algorithm to work, you will need to place at least 1024 images")
            print("of the face you want the model to recognise in the '", name, "' folder")
            print("If you have already placed the images in the folder enter '1'.")
            print("Otherwise, enter '0' and run the program again once the images have been placed.")
            flag_img = int(input())
            if flag_img:
                print("The training process might take a while, but after the model is trained you will be able to use it without training.")
                fr_data = Data()
                fr_data.create_training_data(FR_Training_Dir, FR_Categories, Img_Size, name_x, name_y)
                if(len(fr_data.X) != 0):
                    fr_model = model.initialize_recognition_model(fr_data.X, fr_data.y, name=name)
                    print("The model has been trained to recongise ", name)
                    print("To test the model, run the program again.")
                    print("and place an image in the 'test' folder that is inside the 'FaceRecognition' folder.")
                else:
                    flag_new = 1
                    flag_img = 0
            elif not flag_img:
                print("Please place the images in the corresponding folder and run this program again.")
        elif not flag_new:
            print("To test a model, place an image in the test folder that is inside the 'FaceRecognition' folder.")
            print("The model will give the probability that it is ", name, "'s face. Make sure it is the only image in the 'test' folder.")
            fr_model = model()
            fr_model.initialize_recognition_model([], [], name=name)
            if(fr_model.model != None):
                FR_Testing_Dir = os.path.join(os.path.join("FaceRecognition","test"))
                test_data = Data()
                test_data.create_testing_data(FR_Testing_Dir, Img_Size)
                if (ft_model.predict(test_data.X,batch_size=1) > 0.9):
                    if (ft_model.predict(test_data.X,batch_size=1) > 0.9):
                        print("this is the right face")
                    else:
                        print("this is not thew right face")
                else:
                    print("this is not a face")

if __name__ == "__main__":
    main()



#Variables Explained:
    
#FT_Training_Dir - contains the path of the training data for the face detection model. 
#                  (this variable corresponds to the 'Data_Dir' variable in the Data class)

#FT_Categories - contains a list of size 2, where each element is a string
#                defining one of the labels used by the face detection model. 
#                the model classifies an image as and image of a face or
#                an image of something that is not a face.
#                (this variable corresponds to the 'Categories' variable in the Data class)

#Img_Size - contains an integer value that defines the size of the images used
#           by the models. which means, each image that is sent into the model
#           will have 'Img_Size' pixels on each axis.
#           (this variable corresponds to the 'Img_Size' variable in the Data class)

#ft_data - contains two lists. the first is a list of the processed images
#          that will be used to train the face detection model. the second
#          contains the label of each  image. a label '1' means the image 
#          is a face and a label '0' means it is not.

#ft_model - contains a keras sequential model that is trained to detect
#           the presence of face in an image.

#flag_new - contains 1 or 0. checks if the user wants to train a new face 
#           recognition model or test an existing one.

#name - contains a string that is the name of the new person the user
#       wants to create a new model for. will be used to create a new folder
#       for images of that persons face to train the new model.

#name_x,name_y - both variables contain a string that will be used to create
#              pickle files for the processed images of the new persons face.
#              (these variables correspond to the 'name_x', and 'name_y' variables in the Data class)

#FR_Training_Dir - contains the path of the training data for the face
#                  recognition model of the person the user inputed.
#                  (this variable corresponds to the 'Data_Dir' variable in the Data class)

#FR_Categories - contains a list of size 2, where each element is a string
#                defining one of the labels used by a face recognition model.
#                the model classifies an image as either the face of the
#                person the user inputed or as an image of a face that is not.
#                (this variable corresponds to the 'Categories' variable in the Data class)

#flag_img - contains 1 or 0. checks if the user has placed the images of 
#           the face of the person he wants the model to recognise in
#           the correct folder. if so, the program coninues, if not, it tells
#           the user where exactly to place the images.

#fr_data - contains two lists. the first is a list of the processed images
#          that will be used to train a face recognition model. the second
#          contains the label of each  image. a label '1' means the image 
#          is the face of the person the model is trained to recognise
#          and a label '0' means it is not.

#fr_model - contains a keras sequential model that is trained to recognise
#           if the face of a specific is the face in the image.

#FR_Testing_Dir - contains the path to a folder that will contain a single
#                 image of the face of a specific person. the model will
#                 give its prediction as to whether that face is the one
#                 it has been trained to recognise or not.
#                 (this variable corresponds to the 'Data_Dir' variable in the Data class)

#test_data - contains a numpy array with one element which is the image
#            specified in the description of FR_Training_Dir.


