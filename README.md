# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/visualisation.png "Visualization"
[image2]: ./examples/sign_4.jpg "Limit 70"
[image3]: ./examples/sign_14.jpg "Stop"
[image4]: ./examples/sign_17.jpg "No entry"
[image5]: ./examples/sign_34.jpg "Turn left ahead"
[image6]: ./examples/sign_36.jpg "Continue straight or right"
[image7]: ./examples/sign_confidence.png "Model predictions'confidence"
[image8]: ./examples/sign_4_activations.png "Sign 4 activations"
[image9]: ./examples/sign_14_activations.png "Sign 14 activations"
[image10]: ./examples/sign_17_activations.png "Sign 17 activations"
[image11]: ./examples/sign_34_activations.png "Sign 34 activations"
[image12]: ./examples/sign_36_activations.png "Sign 36 activations"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/gbotev/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Datasets summary

I used basic numpy methods to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

which matches the numbers given in the original [data](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed according to the different labels. As expected train, test and validation data have a common distribution, which means the samples have really been drawn at random. The dataset is not well balanced, as some categories (e.g. 1, 2 - speed limits) have around six times more occurencies than other classes (e.g. 21, 22 - double curve and bumpy road). This might be preferable as the frequency in the dataset resemblance the real life frequencies of these signs on the german roads (double curve and bumpy road are seen less frequently than speed limits). Below is a frequency distribution of the traffic sign classes for train (blue) test (red) and validation (green) datasets

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Data preprocessing (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Only preprocessing was simple data normalization based on substracting 128 and then dividing every pixel by 128.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Fully connected		| input 400 to output 120  						|
| RELU					|												|
| Dropout				| Keep probability 0.5							|
| Fully connected		| input 120 to output 84  						|
| RELU					|												|
| Dropout				| Keep probability 0.5							|
| Fully connected		| input 84 to output 43  						|
| Softmax				| 												|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I have used Adam optimizer because it converges quickly and usually doesn't get stuck in local optima. I used 128 as batch size. I left it the default value as I was not using GPU so I can tune it for maximum speed, and also I think this is a reasonable batch size for the size of our dataset. I have left the learning rate at 0.001 and increased the training epochs to 20, although from around epoch 12-14 there is no substantial improvement in valdiation accuracy.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.955 
* test set accuracy of 0.941

I have used an iterative approach based on the LeNet model. As the time to train the network is relatively small I have increased the epochs to 20 at the beginning. Here are the steps I took:

* First I have trained the LeNet network from the LeNet assignment with unnormalized data resulting in 0.893 validation accuracy.
* By normalizing the data (pixel = (pixel-128) / 128)) I have obtained 0.92 validation accuracy.
* As a next step I increased the conv features in the first layer from 6 to 12, obtaining much better training set accuracy, but almost the same validation accuracy (0.926) meaning the model overfit the data.
* I have tried adding 0.5 dropout on fully-connected 1 and on both fully connected layers resulting in similar performance - I have kept the setting with dropout on both layers at this should improve the generalization. Trying a bit lower and higher drop out rates have resulted in almost the same performance (around 0.96 on validation).
* Finally I have increased the first convolution features from 12 to 16 but this ultimately led to almost the same results as before, so I decided to use the lower number of convolution features.

Obtaining around 0.96 validation accuracy with several different architectures lead me to believe that this is near the best performance we can obtain with the current training data. The data can be augmented with new images, adding noise and flipping the symmetrical signs to obtain more training data and also better results.

The train set accuracy is very high, meaning for the last 6-7 steps we were overfitting the train data, but the validation accuracy has remained stable - the overfitting didn't have a negative effect - and also close to the test set accuracy, so an accuracy around 0.95 is to be expected on examples not yet seen by the model. This would be tested in the next point.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]

The first image might be difficult to classify because it is positioned not in the center, but the upper left corner of the image.
The second image is a stop sign viewed from below, which might be difficult for the model as the proportions would be distorted.
The third and fourth images are photos where the sign takes slightly less pixels than the training data, but using the large patches for convolution layers one and two in addition to the pooling layers should take care of slight differences in pixel size.
The fifth image is a sign that is centered on the image but is slightly different than the analogue in germany, which can be a problem for the network.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 70km/h speed limit	| 30km/h speed limit							| 
| Stop sign   			| Stop sign 									|
| No entry				| No entry										|
| Left turn	      		| Left turn 					 				|
| Go straight or right	| Go straight or right							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of around 95%, given that new images were chosen especially to be challenging. Almost all potential problems were handled by the network. The only difficulty was the highly displaced sign and even then the network got the correct type of sign (speed limit), but the wrong number (50 instead of 70).

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The  code for making predictions on my final model is located in the 20th code cell of the Ipython notebook.

For all the images except the last one the model is pretty sure in its prediction (probability around 0.99). This might be due to the high overfitting of the model on the training data acquiring more than 99% accuracy). The prediction accuracy is also high for the wrongly guessed image, which is a bit disconcerning. The only image that the model is unsure is the last - the sign that suggests to go straight or right. The correct sign is predicted with 88.3% confidence from the model. The second suggestion is a "go right"sign with 11.68% confidence which is explainable given the fact that the picture is not of a german sign and has thicker arrows.

Below is a graph of the confidence that the model assigns to each type of sign with colors red for 70km/h(wrong prediction), green for stop sign, blue for no entry, yellow for left turn and magenta for go straight or right sign.

![alt text][image7]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Here we would examine the first convolution layer activations as it is the easiest to interpret. Below are the activations for the five new signs.
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]

As expected the first conv layer has learned some basic features. For example:
Featuremap 0 detects diagonals sloping to the right (useful for arrows and numbers.
Featuremap 1 and featuremap 2 detect vertical and horizontal lines.
Featuremap 8 has learned to detect red color while featuremap 11 is detecting blue.
Some of the featuremaps have similar activations (e.g. featuremaps 3 and 7, 4 and 10) which means as a further step we can try a lower number of features or some kind of regularization for the first convolution layer.
