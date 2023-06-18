# Deep_learning
Project of Deep Learning: Comparison between classical CNN and a Classifier built with a pre-trained autoencoder

# Abstract 

The project focuses on the comparative study between two different methods of designing convolutional networks. On the one hand, we propose a classically trained convolutional network, while on the other, a classifier built with a pre-trained autoencoder on a non-labeled dataset, to which a dense layer has then been added for classification. Below we present the results obtained, varying the set of labeled data available in order to demonstrate how the accuracy of model 1 (the classic CNN), is better if the labeled data are the majority. On the contrary, it can be observed that the behavior of the network trained with the second method does not seem to have a clear correlation with the data, even if one should expect a trend that tends to decrease as the number of labeled sets increases, this because it has a smaller number to the autoencoder for the pre-train at each step.

# Code 

In final_conv.py you can find whole code for model 1. 
In final_autoencoder.py that for the classifier with autoencoder 
