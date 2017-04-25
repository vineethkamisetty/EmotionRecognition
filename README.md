In this project we have developed a Deep Convolutional Neural Network model for Facial Expression Recognition. Every facial expression is classified into one of the six expressions considered for this project. We implemented convolution-reLU-fully connected layers followed by softmax. To reduce the overfitting of models, we used dropout and local response normalization. To recognize emotions at real time we capture live images from video frames and this image is sent to the model for prediction, then the model outputs the emotion. The test accuracy obtained is 60.1%.<br />

DataSets: FERC-2013(Facial Expression Recognition Challenge) and RaFD(Radboud Faces Database) <br />
FERC contains <br />
i)   28,709 Training images <br />
ii)  3589 Private Test images <br />
iii) 3589 Public Test images <br />

Installation Dependencies: <br />

1) Python 3.5
2) Tensorflow 1.0
3) TFLearn
4) OpenCV 2.0

Tensorflow can either be GPU or CPU version. The model runs faster on a GPU. GPU version of Tensorflow requires additional installations such as CUDA® Toolkit 8.0, NVIDIA drivers associated with CUDA Toolkit 8.0, CuDNN v5.1

It is preferable to install Python through Anaconda as you get all the dependencies installed such as numpy, pandas, matplotlib and other libraries.

Python IDE's : Pycharm or IPython notebook or any other preferred.


File Structure :  <br />

Result_Analysis_images  -----> Contains result and observation figures <br/>
Report_Documents		-----> Folder containing Report, Poster presentation document, block diagrams and flow 								   charts document <br/>
SavedModels             -----> Checkpoints and weights of the saved models <br/>
Models.txt  			-----> Contains tflearn implementation code for models that we have trained <br/> 
Observations.txt 		-----> The observations and results obtained from each model are recorded in this file <br/>
cnn.py                  -----> Architecture of CNN <br/>
cnn_visual.py 			-----> This file is for visualization of the images in between layers of the network <br/>
model_analysis_util.py  -----> Utility functions for generating Prediction matrix, softmax histograms, Top2 accuracies <br/>
recognize_emotion.py 	-----> This is the demo application using OpenCV video capture. It captures video and sends image frames 
							   network model which predicts and outputs the emotions <br/>
util.py                 -----> Support functions for cnn.py <br/>


NOTE : <br />
Folder : ./FercData/    -----> Not uploaded due to size constraint (Need to download from Kaggle) <br />
Folder : ./SavedModels/ -----> All the weights file not uploaded due to size constraint (Need to train all the model to generate weights) <br />

