# Enhancing Jewelry Recognition

## Description:

 In this project, I tried to develop a deep learning-based image classification and object detection system for accurately classifying and detecting 
 jewelry items. By leveraging Convolutional Neural Networks (CNN) for classification and the You Only Look Once (YOLO) algorithm for object detection, 
 this project aims to provide robust solutions for jewelry identification. To showcase the practicality of the developed object detection model, I have 
 integrated it with the PyTorch and OpenCV frameworks. This integration allows for real-time object detection on live camera feeds, enabling users to 
 identify jewelry items instantly.

 The dataset consisted of jewelry images. It was divided into two parts: 
 1.	Training (total count = 1564)
 2.	Test (total count = 250)
 
 Each part had 5 data categories of jewelry:
 1.	Wristwatch  
 2.	Bracelet
 3.	Earrings
 4.	Necklace
 5.	Rings

 This repository contains two directories:
 1. image_clasification: This directory houses the code, zipped dataset, and pickle file of the final model related to the CNN architecture used for image classification.
 2. object_detection_YOLO: Here, you will find the code, zipped labeled dataset, and configuration files for the custom object detection model as well as for real-time object detection. 

 #### Technologies Used:

 * Programming Environment: Google Colab, VS Code
 * Deep Learning Framework: Tensorflow, Keras
 * Computer Vision Library: PyTorch, OpenCV
 * Annotation Tool: LabelImg
 * Pre-trained OD system: YOLO v5


 
 The project encompasses various sections, explained as follow:

 ## Image Classification using CNN Architecture:
 * Open notebook
 * Open dataset

 #### Data Pre-processing:
 
  1. The training dataset was pre-processed by resizing all images to 32x32 pixels and normalizing pixel values between 0 and 1. 

  2. For this purpose, a function was created to first convert an image into an array of pixels of RGB colors having the shape 32x32 using the OpenCV module. Each array and its label (class) in the form of an integer id were appended to a list object.
    
  3. After creating a list of arrays and labels, all elements in the list were shuffled to reduce overfitting in the final model and then two separate lists were created:

     (i)	Feature set ‘X’: list of arrays
     (ii)	The label ‘y’: list of label id 

  5. Both X & y were converted into NumPy arrays, and array X was normalized by dividing each pixel by 255 (maximum pixel value) to scale it in the range 0 to 1.

  6. Similar process was adopted for testing the dataset.

#### Model Architecture:

The CNN model used for this task was built by using Tensorflow’s Keras neural network API. The model had two convolutional layers followed by 
two fully connected layers (Dense layers). Each convolutional layer used a 3x3 filter with a ReLU activation function. Max pooling was applied 
after each convolutional layer with pool size of 2x2. The first convolutional layer had 32 filters, while the last convolutional layers 
had 64 filters. The first fully connected layer had 128 neurons with a ReLU activation function, while the output layer had 5 neurons corresponding 
to the five categories of jewelry with a Softmax activation function.

#### Training Process:

* The model was compiled using the Adam optimizer with accuracy as a metric and sparse categorical cross entropy as a loss function. 
* The model was trained for 10 epochs and a batch size of 32. The model checkpoint was applied to save the epoch with the highest accuracy as the best model. 

#### Results & Discussion:

* The trained model achieved an accuracy of 84.80% on the test set. The Model checkpoint created saved the third epoch as the best model with an accuracy 82.80%. 
* The results show that the CNN model achieved high accuracy on the test set and was able to effectively classify images of Wristwatch, bracelets, Earrings, Necklaces, and Rings.

#### Challenges:

During the course of this task, some challenges were encountered
* Overfitting: Initial model fitting seen high accuracy on the training set but performed poorly on the test set. To address this, the training data of each class was shuffled among each other.
* Optimal number of filters: For convolution layers, a high number of filters was causing overfitting, while a low number of filters was resulting in underfitting. After some trial and error steps, an optimal number of filters was found to be 32 for the first layer and 64 for the second layer.


    


 


 
