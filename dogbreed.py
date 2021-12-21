#import generic libraries               
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import scipy.stats as stats
#import progressbar
import pickle
from time import time

#graphs
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.style as mstyles
import matplotlib.pyplot as mpyplots #plt
#from matplotlib.pyplot import hist
#from matplotlib.figure import Figure

#sklearn
from sklearn.datasets import load_files       

#cuda module
#https://pypi.org/project/opencv-python/
import cv2 

#keras
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.callbacks import ModelCheckpoint  
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

#Tensorflow/keras
#import tensorflow.python_io as tf
#import tensorflow as tf
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.optimizers import SGD
#from keras.applications.resnet50 import ResNet50
#from keras.utils import np_utils
#from keras.layers.core import Dense, Dropout, Activation
#from keras.utils import np_utils

#glob
from glob import glob

#tqm
from tqdm import tqdm

#PIL
from PIL import ImageFile                            

#from extract_bottleneck_features import *

#First part
#from statsmodels.stats import proportion as proptests
#from statsmodels.stats.power import NormalIndPower
#from statsmodels.stats.proportion import proportion_effectsize

#second part
#from scipy.stats import spearmanr
#from scipy.stats import kendalltau
#from scipy.sparse import csr_matrix
#from collections import defaultdict
#from IPython.display import HTML

#setting the random seed
random.seed(8675309)

#altered 2021-12-13
###Dog Breed Project###########s################################################
#########1#########2#########3#########4#########5#########6#########7#########8
def load_dataset(path, 
                 verbose=False):
    '''
    This function loads datasets. Then it splits filenames into into a dataset
    for training input, and targets into categorical dataset for output for our
    Perceptron.

    Special note: this function is STRONGLY based on Udacity notebook for Dog
    Breed Classifications, for completing the Capstone Project for Data Scientist
    course. It may be used for educational purposes only!
    
    Inputs:
    - path (mandatory) - a path for taking a picture - (text string)
    - verbose (optional) - if you want some verbosity under processing
      (default=False)
    
    Output:
    - dog_files
    - dog_targets
    '''
    if verbose:
        print('###function load dataset started')
    
    data = load_files(path)
    
    dog_files = np.array(data['filenames'])
    dog_targets = to_categorical(y=np.array(data['target']), 
                                 num_classes=133)
    
    return dog_files, dog_targets
    
#########1#########2#########3#########4#########5#########6#########7#########8
def face_detector(img_path,
                  distort=False,
                  verbose=False):
    '''
    This function takes an image path and returns a True, if some face could be
    recognized.
    
    Special note: this function is STRONGLY based on Udacity notebook for Dog
    Breed Classifications, for completing the Capstone Project for Data Scientist
    course. It may be used for educational purposes only!
    
    Inputs:
    - img_path (mandatory) - (Text String)
    - verbose (optional) - if you want some verbosity under processing
      (default=False)
    
    Output:
    - True, is a face was recognized in the image (Boolean)
    '''
    if verbose:
        print('###function face detector started')
        
    start = time()
    classifier='haarcascades/haarcascade_frontalface_alt.xml'
    
    #you take an already trained face detector that is taken from a path
    face_cascade = cv2.CascadeClassifier(classifier)
    
    #originally it is a RGB color image
    img = cv2.imread(img_path)
    
    if distort:
        if verbose:
            print('*applying (800x600) distortion emulated')
        img = emulate_svga_dist(image=img,
                                verbose=True)
    if verbose:
        print('*image:', img_path)
    
    #as we seen in the class, normally human faces were converted to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #run face detector method - for grayscale
    faces = face_cascade.detectMultiScale(gray)
    
    #Test function for faces
    if_face = len(faces) > 0 
    num_faces = len(faces)

    #check if it is OK
    if verbose:
        print('*number of faces detected:{}, returning {}'.format(num_faces, if_face))
        
    end = time()
    
    if verbose:
        print('processing time: {:.4}s'.format(end-start))
    
    return if_face, num_faces
    
#########1#########2#########3#########4#########5#########6#########7#########8
def path_to_tensor(img_path,
                   verbose=False):
    '''
    This function takes the path of a image and returns a formatted 4D Tensor.
    
    Special note: this function is STRONGLY based on Udacity notebook for Dog
    Breed Classifications, for completing the Capstone Project for Data Scientist
    course. It may be used for educational purposes only!
    
    Inputs:
    - img_path (mandatory) - (text string)
    - verbose (optional) - if you want some verbosity under processing
      (default=False)

    
    Output:
    A 4-dimensions Tuple, as a Tensor, format (1, 224, 224, 3)
    '''
    if verbose:
        print('###function path to tensor started')

    #RGB image -> PIL.Image.Image
    img = load_img(img_path, 
                   target_size=(224, 224)) #size of the image
    
    #PIL.Image.Image -> 3D tensor dims (224, 224, 3)
    x = img_to_array(img) #3 channels for colors
    
    d4_tensor = np.expand_dims(x, 
                               axis=0) #tensor dims

    return d4_tensor
    
#########1#########2#########3#########4#########5#########6#########7#########8
def paths_to_tensor(img_paths,
                    verbose=False):
    '''
    This function takes images paths and returns it as an Array (rows) of
    Tensors for each one.
    
    Special note: this function is STRONGLY based on Udacity notebook for Dog
    Breed Classifications, for completing the Capstone Project for Data Scientist
    course. It may be used for educational purposes only!
    
    Inputs:
    - img_paths (mandatory)
    - verbose (optional) - if you want some verbosity under processing
      (default=False)

    Output:
    - an Array of stacked Tensors, each one as a vector (each vector is a row)
    
    '''
    if verbose:
        print('###function tensors to array started')
    
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    
    stacked_tensors = np.vstack(list_of_tensors)
    
    return stacked_tensors

#########1#########2#########3#########4#########5#########6#########7#########8
def ResNet50_predict_labels(img_path,
                            distort=False,
                            verbose=False):
    '''
    This function takes an image from a path and runs ResNet50 model to make a
    prediction and returns the index for the best argument.

    Special note: this function is STRONGLY based on Udacity notebook for Dog
    Breed Classifications, for completing the Capstone Project for Data Scientist
    course. It may be used for educational purposes only!

    Inputs:
    - img_path
    - verbose (optional) - if you want some verbosity under processing
      (default=False)

    Output:
    - an Index for the best prediction of an image
    '''
    if verbose:
        print('###function ResNet 50 predictions started')
        
    start = time()

    #creates a prediction for a given image, located in a path
    #OLD way:
    #img = preprocess_input(path_to_tensor(img_path))

    img = load_img(img_path, 
                   target_size=(224, 224))

    if distort:
        if verbose:
            print('*applying (800x600) distortion emulated')
        #creating a distorted image (needs the function below in this notebook)    
        img = emulate_svga_dist(image=img,
                                verbose=True)
        
    x = img_to_array(img) #3 channels for colors    
    d4_tensor = np.expand_dims(x, 
                               axis=0) #tensor dims
    if verbose:
        print('*creating a Tensor from image, with shape:', d4_tensor.shape)
        
    prediction = np.argmax(ResNet50_model.predict(d4_tensor))
    
    end = time()
    
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-start))
    
    return prediction

#########1#########2#########3#########4#########5#########6#########7#########8
def dog_detector(img_path,
                 distort=False,
                 verbose=False):
    '''
    This function returns a True when a dog is detected.
    
    Special note: this function is STRONGLY based on Udacity notebook for Dog
    Breed Classifications, for completing the Capstone Project for Data Scientist
    course. It may be used for educational purposes only!
    
    Inputs:
    - img_path
    - verbose (optional) - if you want some verbosity under processing
      (default=False)
    
    Output:
    - True, if a dog was detected, False else
    '''
    if verbose:
        print('###function dog detector started')
        
    start = time()
    
    if verbose:
        print('*image:', img_path)
        
    prediction = ResNet50_predict_labels(img_path,
                                         distort=distort)
    
    if_prediction = ((prediction <= 268) & (prediction >= 151)) 
    
    if verbose:
        print('*if a dog was detect: {}'.format(if_prediction))
        
    end = time()
    
    if verbose:
        print('processing time: {:.4}s'.format(end-start))
    
    return if_prediction
    
#########1#########2#########3#########4#########5#########6#########7#########8
def VGG16_predict_breed(img_path,
                        verbose=False):
    '''
    This function kahes an image path, process it under VGG16 and return a guess
    for the dog breed.
    
    Special note: this function is STRONGLY based on Udacity notebook for Dog
    Breed Classifications, for completing the Capstone Project for Data Scientist
    course. It may be used for educational purposes only!

    Inputs:
    - img_path
    - verbose (optional) - if you want some verbosity under processing
      (default=False)
    
    Output:
    - a string containing a predicted name for a dog breed
    '''
    if verbose:
        print('###function VGG16 predict breed started')
        
    start = time()
    
    # extract bottleneck features
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    
    # obtain predicted vector
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    
    # return dog breed that is predicted by the model
    predicted_name = dog_names[np.argmax(predicted_vector)]
    
    end = time()

    print('elapsed time: {:.4f}s'.format(end-start))
    
    return predicted_name
    
#########1#########2#########3#########4#########5#########6#########7#########8
def charge_bottlenecks(dic_bottles, 
                       split=False, 
                       architecture=False,
                       filters=32, #arch
                       kernel=4, #arch
                       activation='relu', #arch
                       activation_end='softmax', #arch
                       strides=2, #arch
                       pool=4, #arch
                       padding='same', #arch
                       padding_max='same', #arch
                       model_compile=False,
                       loss_function='categorical_crossentropy', #comp
                       optimizer='rmsprop', #comp
                       summary=False, #comp
                       train=False,
                       epochs=20, #train
                       batch_size=20, #train
                       load=False,
                       test=False,
                       giving='accuracies'):
    '''
    This function takes one or more bottletecks and prepare a complete running.
    
    Special note: this function is STRONGLY based on Udacity notebook for Dog
    Breed Classifications, for completing the Capstone Project for Data Scientist
    course. It may be used for educational purposes only!
    
    It includes:
    0. obtaining the Bottleneck to be completed and runned;
    1. a data split, into Train, Valid and Test datasets;
    2. de definition of the architecture, meaning the completion of the 
       Bottleneck with the necessary final (Dense) layers to run it;
    3. compiling the Model (incluiding an optional Summary visualization);
    4. training the Model;
    5. loading the best trained Model;
    6. testing the model for Accuracy.
    
    Observe that:
    - each step is strongly dependant of the last one. So, you can make only a
      Split, but for doing for example, defining an Architecture, Split will be
      necessary. So, BEFORE turning all these steps True, ensure that the anterior
      steps are are running well. Use it WISELY, or it will crash!
    
    Inputs:
    - dic_bottles (mandatory) - dictionnary with the name of the selected 
      Bottlenecs, with the path for each one - (Dictionnary)
    - split (optional) - add Split phase - (Boolean, default=False)
    - architecture (optional) - add Architecture phase - (Boolean, default=False)
    - filters (optional) - filters for the Dense layer - (Integer, default=32)
    - kernel (optional) - (Boolean, default=4)
    - activation (optional) - (Boolean, default='relu') - activation function
      for the Dense layer (String, default='relu')
    - activation_end (optional) - activation function at the END of the
      Perceptron (String, default='softmax') 
    - padding (optional) - padding option - (Boolean, default='same')
    - model_compile (optional) - add Compile phase - (String, default=False)
    - loss_function (optional) - (Boolean, default='')
    - summary (optional) - add Summary phase - (Boolean, default=False)
    - train (optional) - add Train phase - (Boolean, default=False)
    - epochs (optional) - number of epochs - (Integer, default=20)
    - batch_size (optional) - batches for alleviating the algorithm -
      (Integer, default=20)
    - load (optional) - add Load phase - (Boolean, default=False)
    - test (optional) - add Test phase - (Boolean, default=False)
    - giving (optional) - (String, default='accuracies')
    '''
    print('###function charge bottlekecks started')
        
    start = time()
    
    ls_accuracies = []
    
    for name, link in dic_bottles.items():
        #print(name, link)
        print('{}, preparing charge...'.format(name))
        b_neck_dogb = np.load(link)
        
        if split:
            print('*splitting the data')
            train_dogb = b_neck_dogb["train"]
            valid_dogb = b_neck_dogb["valid"]
            test_dogb = b_neck_dogb["test"]
    
        if architecture:
            print('*defining the architecture')
            dogb_model = Sequential()
            dogb_model.add(Conv2D(filters=filters, 
                                  kernel_size=kernel, 
                                  activation=activation,
                                  strides=strides,
                                  padding=padding, 
                                  input_shape=train_dogb.shape[1:]))
            dogb_model.add(MaxPooling2D(pool_size=pool,
                                        padding=padding_max)) 
            dogb_model.add(GlobalAveragePooling2D()) #GAP layer added!
            dogb_model.add(Dense(133, 
                                 activation=activation_end))                  
            if summary:
                dogb_model.summary()

        if model_compile:
            print('*compiling the model')
            dogb_model.compile(loss=loss_function,
                               optimizer=optimizer,
                               metrics=["accuracy"])            
        if train:
            print('*training the model')
            filepath = 'saved_models/weights.best.dogb.hdf5'
            check = ModelCheckpoint(filepath=filepath, 
                                    verbose=1, 
                                    save_best_only=True)
            dogb_model.fit(train_dogb,
                           train_targets,
                           validation_data=(valid_dogb, valid_targets),
                           epochs=epochs,
                           batch_size=batch_size,
                           callbacks=[check],
                           verbose=1)
            
        if load:
            print('*loading the model')
            dogb_model.load_weights(filepath)
                        
        if test:
            print('*testing the model')
            #first, taking the best prediction
            
            dogb_pred = []
            
            for feature in test_dogb:
                feat_pred = dogb_model.predict(np.expand_dims(feature, axis=0))
                best_pred = np.argmax(feat_pred)
                dogb_pred.append(best_pred)
            
            #second, testing for its Accuracy
            filter_cont = np.array(dogb_pred)==np.argmax(test_targets, axis=1)
            test_accuracy = 100 * (np.sum(filter_cont) / len(dogb_pred))
            print('Test accuracy: {:.4f}'.format(test_accuracy))
            
            ls_accuracies.append((name, test_accuracy))
            
    end = time()

    print('elapsed time: {:.4f}s'.format(end-start))
            
    if giving == 'accuracies':
        return ls_accuracies
        
#########1#########2#########3#########4#########5#########6#########7#########8
def charge_bottlenecks2(dic_bottles, 
                        split=False, 
                        architecture=False,
                        filters=32, #arch
                        kernel=4, #arch
                        activation='relu', #arch
                        activation_end='softmax', #arch
                        strides=2, #arch
                        pool=4, #arch
                        padding='same', #arch
                        padding_max='same', #arch
                        model_compile=False,
                        loss_function='categorical_crossentropy', #comp
                        optimizer='rmsprop', #comp
                        summary=False, #comp
                        train=False,
                        epochs=20, #train
                        batch_size=20, #train
                        load=False,
                        test=False,
                        giving='accuracies'):
    '''
    This function takes one or more bottletecks and prepare a complete running.
    
    Special note: this function is STRONGLY based on Udacity notebook for Dog
    Breed Classifications, for completing the Capstone Project for Data Scientist
    course. It may be used for educational purposes only!
    
    It includes:
    0. obtaining the Bottleneck to be completed and runned;
    1. a data split, into Train, Valid and Test datasets;
    2. de definition of the architecture, meaning the completion of the 
       Bottleneck with the necessary final (Dense) layers to run it;
    3. compiling the Model (incluiding an optional Summary visualization);
    4. training the Model;
    5. loading the best trained Model;
    6. testing the model for Accuracy.
    
    Observe that:
    - each step is strongly dependant of the last one. So, you can make only a
      Split, but for doing for example, defining an Architecture, Split will be
      necessary. So, BEFORE turning all these steps True, ensure that the anterior
      steps are are running well. Use it WISELY, or it will crash!
    
    Inputs:
    - dic_bottles (mandatory) - dictionnary with the name of the selected 
      Bottlenecs, with the path for each one - (Dictionnary)
    - split (optional) - add Split phase - (Boolean, default=False)
    - architecture (optional) - add Architecture phase - (Boolean, default=False)
    - filters (optional) - filters for the Dense layer - (Integer, default=32)
    - kernel (optional) - (Boolean, default=4)
    - activation (optional) - (Boolean, default='relu') - activation function
      for the Dense layer (String, default='relu')
    - activation_end (optional) - activation function at the END of the
      Perceptron (String, default='softmax') 
    - padding (optional) - padding option - (Boolean, default='same')
    - model_compile (optional) - add Compile phase - (String, default=False)
    - loss_function (optional) - (Boolean, default='')
    - summary (optional) - add Summary phase - (Boolean, default=False)
    - train (optional) - add Train phase - (Boolean, default=False)
    - epochs (optional) - number of epochs - (Integer, default=20)
    - batch_size (optional) - batches for alleviating the algorithm -
      (Integer, default=20)
    - load (optional) - add Load phase - (Boolean, default=False)
    - test (optional) - add Test phase - (Boolean, default=False)
    - giving (optional) - (String, default='accuracies')
    '''
    print('###function charge bottlekecks started')
        
    #possible changeable parameters by dictionnary
    ls_par = ['filters', 'kernel', 'activation', 'strides', 
              'padding', 'pool', 'padding_max', 'activation_end']
    ls_acc = [] #best Accuracies attained running the dic machines
    start = time()
    
    #interpreting the dictionnary
    for name in dic_bottles:
        print('{}, preparing charge...'.format(name))
        
        #loading bottleneck (mandatory)
        link=dic_bottles[name]['link']        
        b_neck_dogb = np.load(link)
    
        #changhe parameters (optional)
        parameters=dic_bottles[name]
    
        for key, value in parameters.items():
            if key == 'link':
                print('*link already processed')  
            elif key in ls_par:
                print('*parameter {}="{}", modified by dictionnary'.format(key, value))
                key = value
            else:
                print('*failed {}="{}": this parameter does not exist!'.format(key, value))        
                
        if split:
            print('*splitting the data')
            train_dogb = b_neck_dogb["train"]
            valid_dogb = b_neck_dogb["valid"]
            test_dogb = b_neck_dogb["test"]
    
        if architecture:
            print('*defining the architecture')
            dogb_model = Sequential()
            dogb_model.add(Conv2D(filters=filters, 
                                  kernel_size=kernel, 
                                  activation=activation,
                                  strides=strides,
                                  padding=padding, 
                                  input_shape=train_dogb.shape[1:]))
            dogb_model.add(MaxPooling2D(pool_size=pool,
                                        padding=padding_max)) 
            dogb_model.add(GlobalAveragePooling2D()) #GAP layer added!
            dogb_model.add(Dense(133, 
                                 activation=activation_end))                  
            if summary:
                dogb_model.summary()

        if model_compile:
            print('*compiling the model')
            dogb_model.compile(loss=loss_function,
                               optimizer=optimizer,
                               metrics=["accuracy"])            
        if train:
            print('*training the model')
            filepath = 'saved_models/weights.best.dogb.hdf5'
            check = ModelCheckpoint(filepath=filepath, 
                                    verbose=1, 
                                    save_best_only=True)
            dogb_model.fit(train_dogb,
                           train_targets,
                           validation_data=(valid_dogb, valid_targets),
                           epochs=epochs,
                           batch_size=batch_size,
                           callbacks=[check],
                           verbose=1)
            
        if load:
            print('*loading the model')
            dogb_model.load_weights(filepath)
                        
        if test:
            print('*testing the model')
            #first, taking the best prediction
            dogb_pred = []
            
            for feature in test_dogb:
                feat_pred = dogb_model.predict(np.expand_dims(feature, axis=0))
                best_pred = np.argmax(feat_pred)
                dogb_pred.append(best_pred)
            
            #second, testing for its Accuracy
            filt_pred = np.array(dogb_pred) == np.argmax(test_targets, axis=1)
            test_acc = 100 * (np.sum(filt_pred) / len(dogb_pred))
            
            print('Test accuracy: {:.4f}'.format(test_acc))
            
            ls_acc.append((name, test_acc, dogb_model))
            
    end = time()

    print('elapsed time: {:.4f}s'.format(end-start))
            
    if giving == 'accuracies':
        return ls_acc
        
#########1#########2#########3#########4#########5#########6#########7#########8
def resnet50_dog_pred(model,
                      img_path,
                      verbose=False):
    '''
    This function takes a image by a path and returns a prediction, given a pre
    trained model and his respective Bottleneck.

    Special note: this function is STRONGLY based on Udacity notebook for Dog
    Breed Classifications, for completing the Capstone Project for Data Scientist
    course. It may be used for educational purposes only!

    
    Inputs:
    - model (mandatory) - my pre-trained model goes here
    - img_path (mandatory) - the image path for my prediction
    - verbose (optional) - if you want some verbosity under processing
      (default=False)
    
    Output:
    - best guess for the image at the path
    '''
    if verbose:
        print('###function ResNet 50 dog predictor started')

    start = time()
        
    #First, defining my tensor
    my_tensor = path_to_tensor(img_path)
    
    #Second, extractiong Resnet50 Bottleneck
    b_neck = extract_Resnet50(my_tensor)

    #obtaining my prediction, by running my pre-trained model
    my_pred_vect = model.predict(b_neck)

    #I want only my best prediction, so
    best_pred = dog_names[np.argmax(my_pred_vect)]
    
    end = time()

    if verbose:
        print('elapsed time: {:.4f}s'.format(end-start))
    
    return best_pred
    
#########1#########2#########3#########4#########5#########6#########7#########8
def Image_checker(model,
                  img_path,
                  distort=False,
                  verbose=False):
    '''
    This function takes an image path and checks if it seems as a picture of a 
    dog, or of a human, or neither than any of these categories.
    
    Special note: this function is STRONGLY based on Udacity notebook for Dog
    Breed Classifications, for completing the Capstone Project for Data Scientist
    course. It may be used for educational purposes only!
    
    Inputs:
    - model (mandatory) - the model for classification of breed, if is a human
    - img_path (mandatory) - string of characters for the image path
    - distort (optional) - if you want to emulate the effect of resizing a (3x4)
      format image - (Boolean, default=False)
    - verbose (optional) - if you want some verbosity under processing
      (default=False)
      
    Output:
    - some text about 
      1-if you are recognized as a human
      2-as a dog
      3-neither
      4-if as a human, the dog breed that is more likely at our trained
        classifier
    - if everything runs well, returns True
    '''
    if verbose:
        print('###function image checker started')

    start = time()

    answ1 = face_detector(
                img_path=img_path,
                distort=distort,
                verbose=verbose
    )
    answ2 = dog_detector(
                img_path=img_path,
                distort=distort,
                verbose=verbose
    )
    human = answ1[0]
    dog = answ2
  
    if human:
        print('I detected something that looks like a human')
        
        breed = resnet50_dog_pred(
            model=model,
            img_path=img_path,
            verbose=verbose
        )
        print('...and if you were a dog, you breed should be', breed)
        
        if dog: #this means a bad classification
            print('I also think in someway that it looks like a dog')
            Print('...so please check this image!')
            
    elif dog:
        print('I detected something that looks like a dog')
        if dog: #this means a bad classification
            print('I also think in someway that it looks like a human')
            Print('...so please check this image!')

    else:
        print('Sorry, nothing detected!')
        print('...so please check this image!')
                    
    end = time()

    if verbose:
        print('elapsed time: {:.4f}s'.format(end-start))
        
    return True
    
#########1#########2#########3#########4#########5#########6#########7#########8
def emulate_svga_dist(image,
                      verbose=False):
    '''
    This function takes a 224x224 Perceptron-sized image and applies a
    SVGA-emulated distortion into it.
    
    The idea is that a lot of images came from Webcams and other devices, that
    normally opperates (3x4-camera) proportions. One of the most usual of these
    formats is SVGA (800x600) images. Normally they tells us that Perceptrons
    are robust to slight distortions on the image. So, theoretically the only
    thing you need to do is just to resize this image to 224x244 and all is
    done!
    
    Well, I tried a recognition of my face. It was taken by a VGA Legitech
    webcam, in a well-illuminated office. And my Percetron couldn´t recognize
    me as a man.
    
    So, the idea is to take all these pics that we used to test our Perceptron
    and artificially distort them into a kind of distortion produced when I make
    a horizontal compression of the image.
    
    Inputs:
    - a normal (224x224) RGB .jpg image
    - verbose (optional) - if you want some verbosity under processing
      (default=False)

    Output:
    - an artificially distorted (224x224) RGB .jpg image
    '''
    if verbose:
        print('###function emulate SVGA image distortion (800x600)->(224x224) started')

    start = time()

    h1 = 600
    w1 = 800

    h2 = 224
    w2 = 224

    if verbose:
        print('*working on pre-existing image')

    new_h2 = round(224 * (800 / 600))
    
    if verbose:
        print('*new height for emulating a 800x600 compression:', new_h2)

    #transforming this into an Array
    img_array = img_to_array(image)
    
    if verbose:
        print('*shape of the image 3D Tensor:', img_array.shape)

    #img_dim = (width, height)
    img_dim = (224, new_h2)

    #resized image as array
    img_res = cv2.resize(img_array, 
                         img_dim, 
                         interpolation=cv2.INTER_AREA)

    if verbose:
        print('*new shape of the image 3D RGB Array:', img_res.shape)

    dist_img = array_to_img(img_res)
    
    #img_dim = (x-width, y-height)
    img_dim = (224, 299)

    y_bottom = (new_h2 - 224) // 2

    #using Numpy opperations
    dist_array = np.asarray(dist_img)
    crop_array = dist_array[y_bottom:224+y_bottom, 0:224]

    if verbose:
        print('*new shape after cropping:', crop_array.shape)

    prep_img = array_to_img(crop_array)

    end = time()

    if verbose:
        print('elapsed time: {:.4f}s'.format(end-start))
    
    return prep_img
    
#########1#########2#########3#########4#########5#########6#########7#########8
def decompress_svga_img(image,
                        verbose=False):
    h1 = 600
    w1 = 800

    h2 = 224
    w2 = 224

    if verbose:
        print('*working on pre-existing image')

    #now, I need a new width    
    new_w2 = round(224 * (800 / 600))
    
    if verbose:
        print('*new width for emulating a 800x600 decompression:', new_w2)

    #transforming this into an Array
    img_array = img_to_array(image)
    
    if verbose:
        print('*shape of the image 3D Tensor:', img_array.shape)

    #img_dim = (width, height)
    img_dim = (new_w2, 224)

    #resized image as array
    img_res = cv2.resize(img_array, 
                         img_dim, 
                         interpolation=cv2.INTER_AREA)

    if verbose:
        print('*new shape of the image 3D RGB Array:', img_res.shape)

    dist_img = array_to_img(img_res)
    
    #img_dim = (x-width, y-height)
    img_dim = (299, 224)

    x_bottom = (new_w2 - 224) // 2

    #using Numpy opperations
    dist_array = np.asarray(dist_img)
    crop_array = dist_array[0:224, x_bottom:224+x_bottom,]

    if verbose:
        print('*new shape after cropping:', crop_array.shape)

    prep_img = array_to_img(crop_array)
    
    return prep_img

#altered 2021-11-10
###Convolutional Neural Networks################################################
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_test_accuracy(features_train,
                     targets_train,
                     features_test,
                     targets_test,
                     verbose=False):
    '''This function...
    
    '''
    if verbose:
        print('###funtion test accuracy started')
    start = time()

    #creating a dictionnary for the parameters to be runned
    parameters = {
        'loss_functions': {'categorical_crossentropy', 'mean_squared_error'},
        'activation_functions': {'relu', 'sigmoid'},
        'optimizers': {'rmsprop', 'adam', 'adamax'},
        'layers': {2: {32, 64, 128}},
        'qt_epochs': {50, 100, 150, 200, 250}
    }
    num_layers = 2
    max_acc = 0.
    params = {}

    #training only a 2-layers Perceptron
    for loss_function in parameters['loss_functions']:
        for activation_function in parameters['activation_functions']:
            for optimizer in parameters['optimizers']:
                for epochs in parameters['qt_epochs']:
                    for layers in parameters['layers'][num_layers]:
                        if verbose:
                            print('###Parameter settings')
                            print('loss function:', loss_function)
                            print('activation function:', activation_function)
                            print('optimizer:', optimizer)
                            print('layers:', layers)
                            print('epochs:', epochs)
                    train_acc, test_acc = fn_enhance_model(
                        features_train=features_train,
                        targets_train=targets_train,
                        features_test=features_test,
                        targets_test=targets_test,
                        loss_function=loss_function,
                        activation_function=activation_function,
                        optimizer=optimizer,
                        layers=layers,
                        epochs=epochs,
                        verbose=False
                    )
                    if test_acc > max_acc:
                        max_acc = test_acc
                        params['loss_function'] = loss_function
                        params['activation_function'] = activation_function
                        params['optimizer'] = optimizer
                        params['layers'] = layers
                        params['epochs'] = epochs
    end = time()

    if verbose:
        print('maximum testing accuracy:', max_acc)
        print('for the parameters:')
        print(params)
        print('spent time:', end-start)
        
    return max_acc, params

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_enhance_model(features_train,
                     targets_train,
                     features_test,
                     targets_test,
                     loss_function,
                     activation_function,
                     optimizer,
                     layers,
                     epochs,
                     verbose=False):
    '''This function...
    
    Inputs:
    - features_train
    - targets_train
    - features_test
    - targets_test
    - loss_function
    - activation_function
    - optimizer
    - layers
    - epochs
    - verbose
    
    Outputs:
    -
    '''
    #1.Building the model 
    model = Sequential()
    model.add(Dense(layers, 
                    activation=activation_function, 
                    input_shape=(6,)
    ))
    model.add(Dropout(.2))
    model.add(Dense(layers/2, 
                    activation=activation_function
    ))
    model.add(Dropout(.1))
    model.add(Dense(2, 
                    activation='softmax'
    ))
    #2.Compiling the model
    model.compile(
        loss=loss_function, 
        optimizer=optimizer, 
        metrics=['accuracy']
    )
    if verbose:
        model.summary()
    #3.Training the model
    model.fit(
        features_train, 
        targets_train, 
        epochs=epochs, 
        batch_size=100, 
        verbose=0
    )
    #4.Evaluating the model on the training and testing set
    score = model.evaluate(
        features_train, 
        targets_train
    )
    acc_train = score[1]
    
    if verbose:
        print("\n Training Accuracy:", acc_train)

    score = model.evaluate(
        features_test, 
        targets_test
    )
    acc_test = score[1]
    
    if verbose:
      print("\n Testing Accuracy:", acc_test)

    return acc_train, acc_test
    
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_train(features, 
             targets,
             epochs, 
             learnrate, 
             graph_lines=False,
             verbose=False):
    '''
    This function takes the parameters to train the Perceptron.
    
    This function is strongly based on content from Udacity course Convolutional
    Neural Networks, Lesson 1 - Neural Networks, Class 27 - Gradient Descend and
    may be used only for education purposes.
    
    Inputs:
      - features (mandatory) -
      - targets (mandatory) -
      - epochs (mandatory) -
      - learnrate (mandatory) -
      - graph_lines (optional) - (default=False)
      - verbose (optional) - if you want some verbosity during the process,
        please turn it on (default=False)

    Outputs:
      -
    '''    
    if verbose:
        print('###main train function started')

    start = time()
    errors = []
    n_records, n_features = features.shape
    last_loss = None
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)
    bias = 0
    
    for e in range(epochs):
        if verbose:
            print('for epoch ', e)
        
        del_w = np.zeros(weights.shape)
        
        for x, y in zip(features, targets):
            output = fn_output_formula(
                        features=x,
                        weights=weights, 
                        bias=bias,
                        verbose=verbose
            )
            error = fn_error_formula(
                        y=y, 
                        output=output,
                        verbose=verbose
            )
            weights, bias = fn_update_weights(
                                x=x, 
                                y=y, 
                                weights=weights, 
                                bias=bias, 
                                learnrate=learnrate,
                                verbose=verbose
            )
        
        # Printing out the log-loss error on the training set
        out = fn_output_formula(
                  features=features, 
                  weights=weights, 
                  bias=bias,
                  verbose=verbose
        )
        loss = np.mean(fn_error_formula(
                           y=targets, 
                           output=out,
                           verbose=verbose)
        )
        errors.append(loss)
        
        if e % (epochs / 10) == 0:
            print("\n========== Epoch", e,"==========")
            
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
                
            last_loss = loss
            predictions = out > 0.5
            accuracy = np.mean(predictions == targets)
            print("Accuracy: ", accuracy)
            
        if graph_lines and e % (epochs / 100) == 0:
            m = -weights[0] / weights[1]
            b = -bias / weights[1]
            fn_display(
                m=m, 
                b=b,
                verbose=verbose
            )
            
    # Plotting the solution boundary
    plt.title("Solution boundary")
    m = -weights[0] / weights[1]
    b = -bias / weights[1]
    fn_display(
        m=m, 
        b=b, 
        color='black',
        verbose=verbose
    )

    # Plotting the data
    fn_plot_points(
        X=features, 
        y=targets,
        verbose=verbose
    )
    plt.show()

    # Plotting the error
    plt.title("Error Plot")
    plt.xlabel('Number of epochs')
    plt.ylabel('Error')
    plt.plot(errors)
    plt.show()

    end = time()

    if verbose:
        print('elapsed time: {:.4f}s'.format(end-start))

    return True
    
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_output_formula(features, 
                      weights, 
                      bias,
                      verbose=False):
    '''
    This function takes some parameters and returns them evaluated on a Sigmoid
    function
    
    This function is strongly based on content from Udacity course Convolutional
    Neural Networks, Lesson 1 - Neural Networks, Class 27 - Gradient Descend and
    may be used only for education purposes.
    
    Inputs:
      - features (mandatory) - a list with the features fixed values (Float)
      - weights (mandatory) - a list with the weights of the values (Float)
        *freatures and weights must have the same size!
      - bias (mandatory) - a bias, for dislocation of the zero (if necessary)
      - verbose (optional) - if you want some verbosity during the process,
        please turn it on (default=False)
    Outputs:
      -
    '''
    if verbose:
        print('###function ouput formula started')
        print('*parameters are:')
        print(' weights=', weights)
        print(' bias=', bias)
        
    start = time()
    
    #makes the dot product beweeen features and weights, adding the bias
    #for a final value
    x_val = np.dot(features, weights) + bias
    output = fn_sigmoid(
                 x=x_val,
                 verbose=verbose
    )
    end = time()

    if verbose:
        print('returning:', output)
        print('elapsed time: {:.4f}s'.format(end-start))
    
    return output

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_error_formula(y,
                     output,
                     verbose=False):
    '''
    This function takes a y value and a actual output and returns the y-error.
    It evaluates under Log function.
    
    This function is strongly based on content from Udacity course Convolutional
    Neural Networks, Lesson 1 - Neural Networks, Class 27 - Gradient Descend and
    may be used only for education purposes.
    
    Inputs:
      - y (mandatory) - the y predicted value to be evaluated (Float)
      - output (mandatory) - the actual value, from the model (Float)
      - verbose (optional) - if you want some verbosity during the process,
        please turn it on (default=False)
    Outputs:
      - he difference (the error) between predicted and actual values
    '''
    if verbose:
        print('###function error formula started')
        print('*parameters are:')
        print(' y=', y)
        print(' output=', y)

    start = time()

    error = -y * np.log(output) - (1-y) * np.log(1-output)

    end = time()

    if verbose:
        print('returning:', error)
        print('elapsed time: {:.4f}s'.format(end-start))
    
    return error

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_update_weights(x, 
                      y, 
                      weights,
                      bias,
                      learnrate,
                      verbose=False):
    '''
    This function takes x and y values and updates its values, based on a small
    evaluation rate.
    
    This function is strongly based on content from Udacity course Convolutional
    Neural Networks, Lesson 1 - Neural Networks, Class 27 - Gradient Descend and
    may be used only for education purposes.
    
    Inputs:
      - x (mandatory) - x values for entry
      - y (mandatory) - y values for entry
      - weights (mandatory) - the respective weights
      - learnrate (mandatory) - a fraction of the difference only (so it will be
        a small step, just to not destabilize our model.
      - verbose (optional) - if you want some verbosity during the process,
        please turn it on (default=False)
    Outputs:
      - returns the updated weights for our new model, plus the new bias
    '''
    if verbose:
        print('###function update weights started')
        print('*parameters are:')
        print(' x=', x)
        print(' y=', y)
        print(' weights=', weights)
        print(' bias=', bias)
        print(' learnrate=', learnrate)

    start = time()

    output = fn_output_formula(
                 features=x, 
                 weights=weights, 
                 bias=bias,
                 verbose=verbose,
    )
    d_error = y - output
    weights += learnrate * d_error * x
    bias += learnrate * d_error

    end = time()

    if verbose:
        print('returning new weights and bias')
        print('elapsed time: {:.4f}s'.format(end-start))
    
    return weights, bias

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_plot_points(X,
                   y=None,
                   prepare=False,
                   verbose=False):
    '''
    This is only a plotting function! You give some data (and sometimes some
    parameters) and it plots it for you. In this case, it is a Scatter Plot!
    
    This function is strongly based on content from Udacity course Convolutional
    Neural Networks, Lesson 1 - Neural Networks, Class 27 - Gradient Descend and
    may be used only for education purposes.
    
    Inputs:
      - X (mandatory) - the features (inputs) for your model. Optionally, you
        can set prepare=False and feed it with the raw dataframe
      - y (optional) - the targets (outputs) for your model (if prepare=True,
        you will not provide this dataset!)
      - prepare (optional) - if you take a raw dataset, it will split it X and
        y (default=False)
      - verbose (optional) - if you want some verbosity during the process,
        please turn it on (default=False)
    Outputs:
      - True, if everything goes well
    '''
    if verbose:
        print('###function plot points started')
        print('*parameters are:')
        print(' X=', X)
        print(' y=', y)
    
    if prepare:
        if verbose:
            print('*splitting tge raw dataframe')
        data = X.copy()
        X = np.array(data[["gre","gpa"]])
        y = np.array(data["admit"])

    start = time()

    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    
    plt.scatter(
        [s[0][0] for s in rejected], 
        [s[0][1] for s in rejected], 
        s=25, 
        color='red', 
        edgecolor='k'
    )
    plt.scatter(
        [s[0][0] for s in admitted], 
        [s[0][1] for s in admitted], 
        s=25, 
        color='blue',
        edgecolor='k'
    )

    end = time()

    if verbose:
        print('elapsed time: {:.4f}s'.format(end-start))
    
    return True

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_display(m, 
               b, 
               color='g--',
               verbose=False):
    '''
    This is a plotting function only! It plots a line segment, given parameters
    
    This function is strongly based on content from Udacity course Convolutional
    Neural Networks, Lesson 1 - Neural Networks, Class 27 - Gradient Descend and
    may be used only for education purposes.
    
    Inputs:
      - m (mandatory) - is the m * x parameter for a line
      - b (mandatory) - is the b parameter for a line
      - color (optional) - color for the graph (string, default='g--' -> green)
      - verbose (optional) - if you want some verbosity during the process,
        please turn it on (default=False)
    Outputs:
      - True, if everything runs well
    '''
    if verbose:
        print('###function display started')
        print('*parameters are:')
        print(' m=', m)
        print(' b=', b)
        print(' color=', color)

    start = time()

    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    x = np.arange(-10, 10, 0.1)
    
    plt.plot(
        x,
        m*x+b,
        color
    )
    end = time()

    if verbose:
        print('elapsed time: {:.4f}s'.format(end-start))
    
    return True

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_cross_entropy(Y, 
                     P,
                     verbose=False):
    '''
    This function takes calculates the cross-entropy for model optiomization.
    
    This function is strongly based on content from Udacity course Convolutional
    Neural Networks, Lesson 1 - Neural Networks, Class 21 - Cross Entropy and
    may be used only for education purposes.
    
    Inputs:
      - Y (mandatory) - the error parameter (Integer or Float)
      - P (mandatory) - the probability parameter (Integer of Float)
      - verbose (optional) - if you want some verbosity during the process,
        please turn it on (default=False)
    Outputs:
      - returns the cross_entropy of positive and negative terms
    '''
    if verbose:
        print('###function cross entropy started')

    start = time()

    Y = np.float_(Y)
    P = np.float_(P)
    
    positive = Y * np.log(P)
    negative = (1-Y) * np.log(1-P)
    
    cross_entropy = -np.sum(positive + negative)
    
    end = time()

    if verbose:
        print('elapsed time: {:.4f}s'.format(end-start))

    return cross_entropy

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_softmax1(L,
                verbose=False):
    '''
    This function is a SoftMax evaluation function.
    
    This function is strongly based on content from Udacity course Convolutional
    Neural Networks, Lesson 1 - Neural Networks, Class 16 - Softmax and may be
    used only for education purposes.
    
    Inputs:
      - L (mandatory) - takes a list of elements (Python List)
      - verbose (optional) - if you want some verbosity during the process,
        please turn it on (default=False)
    Outputs:
      - returns a list of evaluated values for each element
    '''
    if verbose:
        print('###function softmax version 1 started')

    start = time()

    expL = np.exp(L)
    sumExpL = sum(expL)
    result = []
    
    for i in expL:
        element = i * 1.0 / sumExpL
        result.append(element)
        
    end = time()

    if verbose:
        print('elapsed time: {:.4f}s'.format(end-start))

    return result

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_softmax2(L,
                verbose=False):
    '''
    This function is the second version of SoftMax.
    
    This function is strongly based on content from Udacity course Convolutional
    Neural Networks, Lesson 1 - Neural Networks, Class 16 - Softmax and may be 
    used only for education purposes.
    
    Inputs:
      - L (mandatory) - a element to be evaluated (Float)
      - verbose (optional) - if you want some verbosity during the process,
        please turn it on (default=False)
    Outputs:
      - an evaluated value (Float)
    '''
    if verbose:
        print('###function softmax version 2 started')

    start = time()

    expL = np.exp(L)
    
    result = np.divide(expL, expL.sum())
    
    end = time()

    if verbose:
        print('elapsed time: {:.4f}s'.format(end-start))

    return result

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_sigmoid(x,
               verbose=False):
    '''This function creates the sigmoid for a power. A sigmoid adjust the power
    value to represent the statistics of being the point in the positive side
    of the boundary, what is the probability to be there

    This function is strongly based on content from Udacity course Convolutional
    Neural Networks, Lesson 1 - Neural Networks, Class 5 - Equation for a line 
    and may be used only for education purposes.

    Inputs:
      - x (mandatory) - the power for the point in your model, according to 
        the boundary conditions (Float)
      - verbose (optional) - if you want some verbosity during the process,
        please turn it on (default=False)
    Output:
      - the sigmoid value (between 0 and 1)
    '''
    if verbose:
        print('###function sigmoid started')

    start = time()

    lower_part = 1 + np.exp(-x)
    sigmoid = 1 / lower_part

    end = time()

    if verbose:
        print('elapsed time: {:.4f}s'.format(end-start))

    return sigmoid

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_stepFunction(t,
                    verbose=False):
    '''This is a step function. If a number is positive, it returns itself. It
    is negative, it returns zero (noise is taken off). It is a very fast
    evaluation method, but not so precise!
    
    This function is strongly based on content from Udacity course Convolutional
    Neural Networks, Lesson 1 - Neural Networks, Class 5 - Equation for a line 
    and may be used only for education purposes.

    Inputs:
      - t (mandatory) - the value for the point, according to your boundary
      - verbose (optional) - if you want some verbosity during the process,
        please turn it on (default=False)
      
    Output:
      - the step value
    '''
    if verbose:
        print('###step function started')

    if t >= 0:
        return 1
    else:
        return 0

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_prediction(X, 
                  W, 
                  b,
                  verbose=False):
    '''This function makes a prediction for a model and gives the step to be
    followed, for the next model, to a new Epoch.
    
    This function is strongly based on content from Udacity course Convolutional
    Neural Networks, Lesson 1 - Neural Networks, Class 5 - Equation for a line 
    and may be used only for education purposes.

    Inputs:
      - X (mandatory) - the features (inputs) for your model
      - W (mandatory) - the W1..Wn parameters for your model
      - b (mandatory) - the bias for the node of your model

    Output:
      - the step value
    '''
    if verbose:
        print('###function prediction started')

    start = time()
    
    #multiplying params for a hyperspace line
    t = (np.matmul(X,W)+b)[0]

    #calling the step function
    prediction = fn_stepFunction(
        t=t,
        verbose=verbose
    )
    
    end = time()

    if verbose:
        print('elapsed time: {:.4f}s'.format(end-start))

    return prediction 

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_perceptronStep(X, 
                      y, 
                      W, 
                      b, 
                      learn_rate=0.01,
                      verbose=False):
    '''This function is the main perceptron step function.
    
    This function is strongly based on content from Udacity course Convolutional
    Neural Networks, Lesson 1 - Neural Networks, Class 5 - Equation for a line 
    and may be used only for education purposes.

    inputs:
      - X (mandatory) - the features (inputs) for your model
      - y (mandatory) - the targets (outputs) for your model
      - W (mandatory) - the W1..Wn parameters for your model
      - b (mandatory) - the bias for the node of your model
      - learn rate (optional) - a small step, for refactoring the whole model
        for best fitting - Float (default=0.01)
      - verbose (optional) - if you want some verbosity during the process,
        please turn it on (default=False)
      
    output:
      - the step value
    '''
    for i in range(len(X)):
        y_hat = fn_prediction(
            X=X[i], 
            W=W, 
            b=b
        )
        if y[i] - y_hat == 1:
            W[0] += X[i][0] * learn_rate
            X[1] += X[i][1] * learn_rate
            b += learn_rate
        elif y[i] - y_hat == -1:
            W[0] -= X[i][0] * learn_rate
            X[1] -= X[i][1] * learn_rate
            b -= learn_rate

    end = time()

    if verbose:
        print('elapsed time: {:.4f}s'.format(end-start))

    return W, b
    
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_trainPerceptronAlgorithm(X, 
                                y,
                                learn_rate=0.01, 
                                num_epochs=25,
                                verbose=False):
    '''This function runs the perceptron algorithm repeatedly on the dataset,
    and returns a few of the boundary lines obtained in the iterations,
    for plotting purposes. Feel free to play with the learning rate and the 
    num_epochs, and see your results plotted.
    
    This function is strongly based on content from Udacity course Convolutional
    Neural Networks, Lesson 1 - Neural Networks, Class 5 - Equation for a line 
    and may be used only for education purposes.

    inputs:
      - X (mandatory) - the features (inputs) for your model
      - y (mandatory) - the targets (outputs) for your model
      - learn rate (optional) - a small step, for refactoring the whole model
        for best fitting - Float (default=0.01)
      - num_epochs (optional) - number of steps for refactoring the model - 
        Integer (default=25)
        *laterly it can be added an alternative breaking condition, that the
         error is lower than a predetermined value
      - verbose (optional) - if you want some verbosity during the process,
        please turn it on (default=False)
      
    output:
      - boundary lines for the trained Perceptron
    '''

    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    
    # These are the solution lines that get plotted below.
    boundary_lines = []
    
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = fn_perceptronStep(
                   X=X, 
                   y=y, 
                   W=W, 
                   b=b, 
                   learn_rate=learn_rate
        )
        boundary_lines.append((-W[0]/W[1], -b/W[1]))

    end = time()

    if verbose:
        print('elapsed time: {:.4f}s'.format(end-start))

    return boundary_lines
