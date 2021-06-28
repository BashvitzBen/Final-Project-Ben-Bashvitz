# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 10:13:33 2021

@author: bashv
"""


from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Activation, Flatten, MaxPooling2D

class model():
    #this class handles all functions that relate to creating, training and saving the models used in the program.
    model = None #variable model will contain the finished keras sequential model.
    
    def __init__(self):
        #initializes the model vaiable of the class as a keras sequential model.
        self.model = Sequential()
        
    def initialize_face_detection_model(self,X,y):
        #this function checks if the face detection model has already been trained. if it has been, it loads it into variable model.
        #if it has not, it adds layers to the model variable and saves it so it can later be loaded in.
        #input: two arrays which contain the processed images and their labels. more information in Variables Explained below.
        #output: adds all the layers needed for the model and saves it. also prompts the user with a few messages to make sure
        #        he understands what is currently happening.
        try:
            self.model = load_model("ft_model")
            print("The face detection model has already been trained, you may proceed.")
        except Exception as e:
        
            print("The data is now loaded and the face detection model will start it's training process.")
            print("The progress of this procedure will be shown in the console.")
            
            self.add_convolutional_Layer(64, (3,3), input_shape = X.shape[1:])
            self.add_convolutional_Layer(64, (3,3), input_shape = X.shape[1:])
            self.add_densing_layers(64)
            self.train_model(X, y, 32, 10, 0.1)
            self.model.save("ft_model")
        
            print("The face detection model has finished it's training.")
        
    
    def initialize_recognition_model(self,X,y,name):
        #this function checks if a face recognition model for 'name' has already been trained. it it has, it loads it into variable model.
        #if it has not, it adds layers to the model variable and saves it so it can later be loaded in.
        #input: two arrays which contain the processed images and their labels
        #       and a string name that corresopnds to the person the user wanted the model to recognise more information in Variables Explained below.
        #output: adds all the layers needed for the model and saves it. also prompts the user with a few messages to make sure he understands
        #        what is currently happening.
        try:
            self.model = load_model("fr_model_"+name)
            print("The face recognition model for ",name," has already been trained, you may proceed.")
        except Exception as e:
            if(len(X) != 0):
                print("The data is now loaded and the face recognition model will start it's training process.")
                print("The progress of this procedure will be shown in the console.")
            
                self.add_convolutional_Layer(32,(3,3),input_shape=X.shape[1:])
                self.add_convolutional_Layer(64,(3,3),input_shape=X.shape[1:])
                self.add_convolutional_Layer(128,(3,3),input_shape=X.shape[1:])
                self.add_densing_layers(128)
                self.train_model(X, y, 32, 5, 0.1)
                self.model.save("fr_model_"+name)
            
                print("The face recognition model has finished it's training.")
            else:
                self.model = None
                print("a model for this person does not exist, the program will end now.")
                print("make sure you train a model for this person before you test it.")
       
    
    def add_convolutional_Layer(self,num_filters,shape_filters,input_shape):
        #this function adds the following layer to the model variable: a 2d convolutional layer,
        #an activation function for that convolutional layer and a 2d max pooling layer.
        #input: integer num_filters which corresponds to the number of weights added in the convolutional layer.
        #       tuple shape_filters which corresponds to the shape of the weights used by the convolutional layer.
        #       tuple input_shape which tells the convolutional layer what is the size of the images it will work on.
        #output: adds a convolutional layer to the model
        self.model.add(Conv2D(num_filters,shape_filters,input_shape = input_shape))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
    
    def add_densing_layers(self,channels):
        #this function adds a flattening layer to the model and two densing layers and a sigmoid activation function. this will make it so the output
        #of the model is a single number between 0 and 1. where 1 is a positive result and 0 is a negative result.
        #input: an integer that describes the number of channels of the output of the last convolutional layer. it is used in the densing function to dense the output
        #       into size 'channels'.
        #output: adds a flattening layer, two densing layers and a sigmoid activation function to the model.
        self.model.add(Flatten())
        self.model.add(Dense(channels))
        self.model.add(Dense(1))
        self.model.add(Activation("sigmoid"))
    
    def train_model(self, X, y, batch_size, epochs, validation_split):
        #this function compiles the model with the right loss function, 
        #optimizers and metrics in order to increase the accuracy of the model's predictions.
        #then, it trains the model using the inputed variables.
        #input: two arrays which contain the processed images and their labels.
        #       two integers batch_size and epochs which tells the training function
        #       how many images to fit per batch and how many times it need to repeat that process.
        #output: compiles and trains the model variable.
        self.model.compile(loss="binary_crossentropy", optimizer = "adam", metrics=["accuracy"])
        self.model.fit(X,y,batch_size = batch_size,epochs = epochs, validation_split = validation_split)




