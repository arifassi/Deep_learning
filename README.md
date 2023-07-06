# Deep_learning
Project of Deep Learning: Comparison between classical CNN and a Classifier built with a pre-trained autoencoder

# Abstract 

The project focuses on the comparative study between two different methods of designing convolutional networks. On the one hand, we propose a classically trained convolutional network, while on the other, a classifier built with a pre-trained autoencoder on a non-labeled dataset, to which a dense layer has then been added for classification. Below we present the results obtained, varying the set of labeled data available in order to demonstrate how the accuracy of model 1 (the classic CNN), is better if the labeled data are the majority. On the contrary, it can be observed that the behavior of the network trained with the second method does not seem to have a clear correlation with the data, even if one should expect a trend that tends to decrease as the number of labeled sets increases, this because it has a smaller number to the autoencoder for the pre-train at each step.

# Code 

This project has two version: the second it is focused on decresing the ratio, in order to show the drop in the accuracy of the CNN trained normally and the difference by introducing it in the autoencoder classifier. 

# Authors 
Arianna Fassino e
Francesco Devecchi 
