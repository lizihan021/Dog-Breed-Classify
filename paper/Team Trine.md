Team Trine
===

Group member: Lina Zhang, Wen He, Zihan Li

#### 1. Objective & Dataset:

We want to classify dogs' breed, which is fine-grained classification. 

For dataset, our **first choice** is [**Columbia dogs with parts dataset**](https://people.eecs.berkeley.edu/~kanazawa/). 133 breeds, 8,351 images of dogs and 8 part locations annotated for each image.

Our second choice for dataset is [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/). There are 20580 dogs belonging to 120 breeds in the data set. (Just incase there is some issue with the first one, though unlikely).

#### 2. Approach & Related Work:

We will first use **CNN** and labeled data to learn the position of dog's face and interesting points on the face. After that we will construct **SIFT feature vectors** of the interesting points. After that we will classify the dogs breed based on a rbf kernel **SVM** with SIFT feature vectors as input. 

Based on this paper: [Liu et al.](https://people.eecs.berkeley.edu/~kanazawa/papers/eccv2012_dog_final.pdf) 

There is no code found on the internet, so we will write and train our own program.