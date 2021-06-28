# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 10:29:15 2021

@author: bashv
"""

import numpy as np #for creating arrays and performing functions on them.
import os #for creating path to files and directories.
import cv2 #for loading in images.
import random #for shuffling arrays at random.
import pickle #for saving and loading data in special pickle files.


class Data():
    
    X = None #a numpy array that will contain the processed images.
    y = None #a numpy array that will contain the labels for the images. label '1' for a positive result and lbel '0' for a negative result.
    
    def __init__(self):
        #this function initializes the X and y lists which later on would become numpy arrays that contain the processed data.
        self.X = []
        self.y = []
    
    def create_training_data(self,Data_Dir, Categories, Img_Size, name_x, name_y):
        #this function inputs the images and their labels from either the folder with path 'Data_Dir' 
        #or from the pickle files with names 'NameX' and 'NameY'
        #input: all of the variables this function takes as inputs
        #       are explained it the main function in the 'Face Recognition.py' file.
        #       please refer to that explanation.
        #output: updates the X and y variables to contain the numpy arrays
        #        specified in the explnation in lines 17 and 18.
        #        also prompts the user about the process of loading in the images and labels.
                 
        training_data = []
        try:
            pickle_in = open(name_x,"rb") 
            self.X = pickle.load(pickle_in)
            pickle_in = open(name_y,"rb")
            self.y = pickle.load(pickle_in) 
            
        except Exception as e1:
            print("It seems this is the first time you are running this program. Before you continue,")
            print("place the 'Face Detection' and 'FaceRecognition' folder inside the project folder.")
            print("if you have already placed the folders in the project folder enter '1'.")
            print("Otherwise, enter '0' and run the program again once you have done so.")
            flag = int(input())
            if flag:
                print("The program is now loading the data for the face detection algorithm for the first time. this process might take a while.")
                print("When this process is finished you will be prompted with more messages.")
                for category in Categories: #this for loop iterates over each category in list categories (which is 2 categories)
                    path = os.path.join(Data_Dir, category)
                    class_num = Categories.index(category)
                    for img in os.listdir(path): #this for loop iterates over each image in directory 'Data_Dir'\'category'
                                                 #and adds it to the training_data array after it resizes it.
                        try:
                            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                            new_array = cv2.resize(img_array,(Img_Size,Img_Size))
                            training_data.append([new_array,class_num])
                        except Exception as e:
                            pass            
            if(len(training_data) == 0):
                print("please make sure you have placed the images in the right place and run the program again.")
            else:
                random.shuffle(training_data)
                print(len(training_data))
        
                for features,label in training_data:
                    self.X.append(features)
                    self.y.append(label)

                self.X = np.array(self.X).reshape(-1,Img_Size,Img_Size,1)
                self.X = self.X/255.0
                self.y = np.array(self.y)
                pickle_out = open(name_x,"wb")
                pickle.dump(self.X,pickle_out)
                pickle_out.close()

                pickle_out = open(name_y,"wb")
                pickle.dump(self.y,pickle_out)
                pickle_out.close()
        pass

    def create_testing_data(self, Data_Dir, Img_Size):
        #this function loads the image placed in the 'FaceRecognition\test' Directory
        #and processes it for the face recognition model to predict.
        #input: all of the variables this function takes as inputs
        #       are explained it the main function in the 'Face Recognition.py' file.
        #       please refer to that explanation.
        for img in os.listdir(Data_Dir):
            img_array = cv2.imread(os.path.join(Data_Dir,img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array,(Img_Size,Img_Size))
        self.X.append(np.array(new_array).reshape(-1,Img_Size,Img_Size,1))
        self.X = self.X/255.0
        
        

#Variables Explained:

#pickle_in - contains information on which pickle file to open.

#flag - either 1 or 0. checks if the user has placed the 'Face Detection' 
#       and 'Face Recognition' folder in the project folder.

#path - contains the path to the directory where the images used by the model
#       are placed.

#class_num - contains the label (either 1 or 0) of the current category of images.

#img_array - contains an image from the directory from 'path'.

#new_array - contains a resize image which is now in size
#            'Img_Size' by 'Img_Size'.

#training_data - contains an array where every element
#                is a list with two elements, the first element
#                being the processed image and the second being it's label.

#pickle_out - contains information needed to save the processed data
#             in a pickle file.