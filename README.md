# ANN_Noise_Study
Repository containing code and data files relating to the study of update patterns in ANNS (Artificial Neural Networks).

## Background
This all started with looking at the heatmap of the weights of a Neural Network. The heatmap looks seemingly like random noise, but then I compared the updated heatmap to the heatmap of the network weights when starting out, and the intensities of the heatmap on the trained one, while duller in comparison to the pre-trained heatmap, contained similar intensity patterns to the post-training heatmap. This repository aims to explore that relationship.

## Goals
The main goals of this project are to explore why Neural Networks follow this update pattern, and possibly gain some insight into common problems of ANNs such as the catastrophic forgetting problem.

## Testing Environment
As of this update, I have only used the basic MNIST dataset and PyTorch, mainly because it was the quickest to get setup and testing. I have also only tested thus far on Sigmoid and ReLU models, and have plans to test more models in the future.


