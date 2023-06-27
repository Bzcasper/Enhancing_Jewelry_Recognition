# Enhancing Jewelry Recognition


## Description

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
 

 ## Image Classification using CNN Architecture
 * [Open notebook](https://github.com/Pushpalal/Enhancing_Jewelry_Recognition/blob/ed729bd9400ad0145b94f986349fe384c5fa64ff/image_classification/jewellery_class_cnn.ipynb)
 * [Open dataset](https://github.com/Pushpalal/Enhancing_Jewelry_Recognition/blob/ed729bd9400ad0145b94f986349fe384c5fa64ff/image_classification/img_data_for_class.zip)

 #### Data Pre-processing:
 
  1. The training dataset was pre-processed by resizing all images to 32x32 pixels and normalizing pixel values between 0 and 1. 

  2. For this purpose, a function was created to first convert an image into an array of pixels of RGB colors having the shape 32x32 using the OpenCV module. Each array and its label (class) in the form of an integer id were appended to a list object.
    
  3. After creating a list of arrays and labels, all elements in the list were shuffled to reduce overfitting in the final model and then two separate lists were created:

     (i)	Feature set ‘X’: list of arrays.
     (ii)	The label ‘y’: list of label id. 

  5. Both X & y were converted into NumPy arrays, and array X was normalized by dividing each pixel by 255 (maximum pixel value) to scale it in the range 0 to 1.

  6. Similar process was adopted for testing the dataset.

#### Model Architecture:

The CNN model used for this task was built by using Tensorflow’s Keras neural network API. The model had two convolutional layers followed by 
two fully connected layers (Dense layers). Each convolutional layer used a 3x3 filter with a ReLU activation function. Max pooling was applied 
after each convolutional layer with a pool size of 2x2. The first convolutional layer had 32 filters, while the last convolutional layers 
had 64 filters. The first fully connected layer had 128 neurons with a ReLU activation function, while the output layer had 5 neurons corresponding 
to the five categories of jewelry with a Softmax activation function.

#### Training Process:

* The model was compiled using the Adam optimizer with accuracy as a metric and sparse categorical cross entropy as a loss function. 
* The model was trained for 10 epochs and a batch size of 32. The model checkpoint was applied to save the epoch with the highest accuracy as the best model. 

#### Results & Discussion:

* The trained model achieved an accuracy of 84.80% on the test set. The Model checkpoint created saved the third epoch as the best model with an accuracy of 82.80%. 
* The results show that the CNN model achieved high accuracy on the test set and was able to effectively classify images of Wristwatch, bracelets, Earrings, Necklaces, and Rings.

#### Challenges:

During the course of this task, some challenges were encountered
* Overfitting: Initial model fitting seen high accuracy on the training set but performed poorly on the test set. To address this, the training data of each class was shuffled among each other.
* Optimal number of filters: For convolution layers, a high number of filters was causing overfitting, while a low number of filters was resulting in underfitting. After some trial and error steps, an optimal number of filters was found to be 32 for the first layer and 64 for the second layer.


## Custom Object Detection using YOLO v5

I have utilized the YOLO v5 algorithm to train a custom object detection model specifically tailored for the jewelry dataset. It is an object detection algorithm written in the PyTorch framework. I used LabelImg to label images. It is an image annotation tool, which is written in Python and uses Qt for its graphical interface. For more information related to the installation and use of LabelImg, refer to this [link](https://youtu.be/fjynQ9P2C08).

* [Labeled dataset using LabelImg](https://github.com/Pushpalal/Enhancing_Jewelry_Recognition/blob/ed729bd9400ad0145b94f986349fe384c5fa64ff/object_detection_YOLO/labeled_data_for_OD.zip)
* [Open notebook](https://github.com/Pushpalal/Enhancing_Jewelry_Recognition/blob/ed729bd9400ad0145b94f986349fe384c5fa64ff/object_detection_YOLO/develop_OD_model_YOLOv5.ipynb)

#### Training:

1.	YOLO v5 trained model repository was cloned on the Google Colab environment using the git clone command, which made available YOLOv5 code and pre-trained weights in the Colab environment. After that, all required dependencies were also installed using the requirements.txt file. 

2.	Google Drive was mounted to install the labeled dataset files at the required path. 

3.	The 'dataset.yaml' file was modified to update the train path, test path, numbers of classes, and class names.

4.	Finally, the model was trained for 50 epochs with a batch size of 16 and an image size of 640x640.

#### Results:
The performance of the trained model was evaluated on the test dataset using the mean Average Precision (mAP) metric. The mAP obtained was 0.833. Among all five classes, Wristwatch was having the highest precision of 0.995 and the Necklace was having the lowest precision of 0.544. 


## Real-time Object Detection

Finally, I created a Python program to run real-time jewelry detection on a webcam. I used PyTorch to load the trained [model](https://github.com/Pushpalal/Enhancing_Jewelry_Recognition/blob/ed729bd9400ad0145b94f986349fe384c5fa64ff/object_detection_YOLO/best.pt), which was developed earlier. The OpenCV (cv2) module was used for video capture, frame manipulation, display, and input event handling.

* [Open code file](https://github.com/Pushpalal/Enhancing_Jewelry_Recognition/blob/ed729bd9400ad0145b94f986349fe384c5fa64ff/object_detection_YOLO/realtime_jewellery_detection_OnCam.py)
* [Requirements](https://github.com/Pushpalal/Enhancing_Jewelry_Recognition/blob/ed729bd9400ad0145b94f986349fe384c5fa64ff/object_detection_YOLO/requirements_for_realtime_detection.txt) for running above Python program on virtual environment.
* [Trained Model](https://github.com/Pushpalal/Enhancing_Jewelry_Recognition/blob/ed729bd9400ad0145b94f986349fe384c5fa64ff/object_detection_YOLO/best.pt) developed for OD.


## Future Work

* Scaling the model to handle larger jewelry datasets and a wider range of jewelry categories.
* Investigating additional image preprocessing techniques to further enhance model performance.
* Exploring more advanced object detection algorithms or frameworks for improved accuracy and speed.




 


 
