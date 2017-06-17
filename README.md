# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, I take a video stream and use a convolutional neural network from the paper
You Only Look Once and detect cars found in the stream.

[//]: # (Image References)
[image1]: ./examples/yolo_sgrid.png
[image2]: ./examples/bboxes1.PNG
[image3]: ./examples/bboxes2.PNG
[image4]: ./examples/bboxes3.PNG
[video1]: ./video_output.mp4


Regression v.s. Classification :
--------------------------------
Typically when detecting an object in a video stream there are two methodologies, the first consists using a classifier such as a
Support Vector Machine (SVM) trained on a set of images whose features have been extracted using a
Histogram of Oriented Gradients (HOG) which is then used to classify different portions of an image using
 a sliding window
method. The second involves treating the problem as a regression problem such as using a Single Shot Multibox Detector(SSD)
 or You Only Look Once (YOLO) network. In this case I have decided to treat the problem as regression problem
 because this allows one to directly take the pixels from an image and output predicted bounding boxes and probabilities.
 Using a high end GPU this allows for a reasonably fast and accurate network.

The Project
---

The goals / steps of this project are the following:

* Building Keras model of YOLO
* Load pre-trained weights
* Preprocess image to be input into network
* Postprocess tensor output from network
* Build detection pipeline
* Apply to video Stream

The Network
-----------
The chosen network for this project is based on tiny YOLO v1. This networks architecture consists
 12 layers, 9 convolution layers followed by three fully connected layers all with Leaky RelU
  activation functions. Below is a summary of the models architecture

   ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to
    ====================================================================================================
    convolution2d_1 (Convolution2D)  (None, 16, 448, 448)  448         convolution2d_input_1[0][0]
    ____________________________________________________________________________________________________
    leakyrelu_1 (LeakyReLU)          (None, 16, 448, 448)  0           convolution2d_1[0][0]
    ____________________________________________________________________________________________________
    maxpooling2d_1 (MaxPooling2D)    (None, 16, 224, 224)  0           leakyrelu_1[0][0]
    ____________________________________________________________________________________________________
    convolution2d_2 (Convolution2D)  (None, 32, 224, 224)  4640        maxpooling2d_1[0][0]
    ____________________________________________________________________________________________________
    leakyrelu_2 (LeakyReLU)          (None, 32, 224, 224)  0           convolution2d_2[0][0]
    ____________________________________________________________________________________________________
    maxpooling2d_2 (MaxPooling2D)    (None, 32, 112, 112)  0           leakyrelu_2[0][0]
    ____________________________________________________________________________________________________
    convolution2d_3 (Convolution2D)  (None, 64, 112, 112)  18496       maxpooling2d_2[0][0]
    ____________________________________________________________________________________________________
    leakyrelu_3 (LeakyReLU)          (None, 64, 112, 112)  0           convolution2d_3[0][0]
    ____________________________________________________________________________________________________
    maxpooling2d_3 (MaxPooling2D)    (None, 64, 56, 56)    0           leakyrelu_3[0][0]
    ____________________________________________________________________________________________________
    convolution2d_4 (Convolution2D)  (None, 128, 56, 56)   73856       maxpooling2d_3[0][0]
    ____________________________________________________________________________________________________
    leakyrelu_4 (LeakyReLU)          (None, 128, 56, 56)   0           convolution2d_4[0][0]
    ____________________________________________________________________________________________________
    maxpooling2d_4 (MaxPooling2D)    (None, 128, 28, 28)   0           leakyrelu_4[0][0]
    ____________________________________________________________________________________________________
    convolution2d_5 (Convolution2D)  (None, 256, 28, 28)   295168      maxpooling2d_4[0][0]
    ____________________________________________________________________________________________________
    leakyrelu_5 (LeakyReLU)          (None, 256, 28, 28)   0           convolution2d_5[0][0]
    ____________________________________________________________________________________________________
    maxpooling2d_5 (MaxPooling2D)    (None, 256, 14, 14)   0           leakyrelu_5[0][0]
    ____________________________________________________________________________________________________
    convolution2d_6 (Convolution2D)  (None, 512, 14, 14)   1180160     maxpooling2d_5[0][0]
    ____________________________________________________________________________________________________
    leakyrelu_6 (LeakyReLU)          (None, 512, 14, 14)   0           convolution2d_6[0][0]
    ____________________________________________________________________________________________________
    maxpooling2d_6 (MaxPooling2D)    (None, 512, 7, 7)     0           leakyrelu_6[0][0]
    ____________________________________________________________________________________________________
    convolution2d_7 (Convolution2D)  (None, 1024, 7, 7)    4719616     maxpooling2d_6[0][0]
    ____________________________________________________________________________________________________
    leakyrelu_7 (LeakyReLU)          (None, 1024, 7, 7)    0           convolution2d_7[0][0]
    ____________________________________________________________________________________________________
    convolution2d_8 (Convolution2D)  (None, 1024, 7, 7)    9438208     leakyrelu_7[0][0]
    ____________________________________________________________________________________________________
    leakyrelu_8 (LeakyReLU)          (None, 1024, 7, 7)    0           convolution2d_8[0][0]
    ____________________________________________________________________________________________________
    convolution2d_9 (Convolution2D)  (None, 1024, 7, 7)    9438208     leakyrelu_8[0][0]
    ____________________________________________________________________________________________________
    leakyrelu_9 (LeakyReLU)          (None, 1024, 7, 7)    0           convolution2d_9[0][0]
    ____________________________________________________________________________________________________
    flatten_1 (Flatten)              (None, 50176)         0           leakyrelu_9[0][0]
    ____________________________________________________________________________________________________
    dense_1 (Dense)                  (None, 256)           12845312    flatten_1[0][0]
    ____________________________________________________________________________________________________
    dense_2 (Dense)                  (None, 4096)          1052672     dense_1[0][0]
    ____________________________________________________________________________________________________
    leakyrelu_10 (LeakyReLU)         (None, 4096)          0           dense_2[0][0]
    ____________________________________________________________________________________________________
    dense_3 (Dense)                  (None, 1470)          6022590     leakyrelu_10[0][0]
    ====================================================================================================
    Total params: 45,089,374
    Trainable params: 45,089,374
    Non-trainable params: 0
    ____________________________________________________________________________________________________

The models initial convolution layers extract the features from the image,
and the fully connected layers
make the predictions for the bounding boxes and class probabilities.
The model was trained on PASCAL Visual Object Classes and the pre-trained weights were used.

Pre-processing
--------------

The image pre-processing consisted of first cropping the image to avoid running the network unnecessarily on locations where there would be no cars
 such as in the sky. After the image is cropped, the image is then resized as the network requires the input to have a size of 448 by 448. Then a
 simple image normalization is performed giving each pixel a value between -1 and 1. Lastly the image is transformed to have a shape of (channel,height,width)
 as is required by the model.

Tensor output
-------------
The model works by dividing the image into an S x S grid and for each grid cell predicts the bounding boxes
B, the confidence for those boxes
and class probabilities C.
These predictions are encoded into a tensor of length S*S*(B*5+C) which is the output of the network.
Because the network was trained on PASCAL VOC S=7 for a 7 by 7 grid, B=2 for two possible bounding box predictions in each grid cell
, and C=20 for the twenty possible classes in the data.
This results in a tensor of length 1470 and has the following structure, the first 980 values are the probabilities
for each grid cell for the corresponding 20 classes. The next 98 values are the confidence scores
 for the two predicted bounding boxes in each cell.
The last 392 values are the encoded bounding box coordinates (x,y,w,h)

![alt text][image1]

Post-processing
---------------
After running the images through the model, the output tensor requires significant post processing to extract all of the stored information about
the predicted classes and bounding boxes. Details pertaining to this extraction of information can be found in `detection_pipeline.ipynb` under the post processing heading

Model Results
-------------
Using a threshold of 0.20 we get the following images from the model ran on the test images

![alt text][image2]

![alt text][image3]

![alt text][image4]

[Here][video1] are the result found when the detection pipeline is also applied to the project_video.mp4

Discussion
----------
Overall I was happy with the way this project turned out, the model is pretty fast resulting in almost real time results taking only 1 minute and 6 seconds on a
50 second video when ran on a gtx 1070. Had a better graphics card been used such as a 1080 or Titan Xp then we could probably use this model in real time.
Some issues found with the pipeline is that it would sometimes lose track of the car for a few moments, which could certaintly cause issues, thus the accuracy of the
network could be improved.

References
----------
- [You Only Look Once (YOLO) paper](https://arxiv.org/abs/1506.02640)
- [Darknet](https://github.com/pjreddie/darknet)
- [Darknet to Keras (YAD2K)](https://github.com/allanzelener/YAD2K)
- [Xslittlegrass Implemintation](https://github.com/xslittlegrass/CarND-Vehicle-Detection)
- [Subodh Malgonde Implemination](https://github.com/subodh-malgonde/vehicle-detection)

