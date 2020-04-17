## Multi-Classifier-Deep-Neural-Network-for-Detecting-Intruder-behavior-in-Cyber-Security

**This repo contains all files required and the steps for the MCDNN  model implementation.**

This is the  [paper](https://ieeexplore.ieee.org/document/9067548/authors#authors) and you can get its free pdf version [here](https://github.com/Desire100/Multi-Classifier-Deep-Neural-Network-for-Detecting-Intruder-behavior-in-Cyber-Security-/blob/master/Paper.pdf). It was published by [IEEE](https://ieeexplore.ieee.org/document/9067548/authors#authors) 

#### Citation 

D. Iradukunda, X. W. An and W. Ali, "Multi-Classifier Deep Neural Network for Detecting Intruder Behavior In Cyber Security," 2019 16th International Computer Conference on Wavelet Active Media Technology and Information Processing, Chengdu, China, 2019, pp. 373-377.

### Paper Abstract


The security is becoming much more important due
to the massive growth in computer network technologies. Finding
anomalies from a system’s network traffic has been a hot research
topic over the last two decades. Most of the existing signature-
based intrusion detection systems only rely upon the pre-
configured and predetermined attack patterns that are practically
not true for real life challenges. In this research work, we attempt
to capture the dynamic behavior of intruders by utilizing a novel
Multi-Classifier Deep Neural Network (MCDNN) framework.
The proposed framework intuitively segregate the attackers and
authorize user data packets for a given network traffic. The
MCDNN is based on four layers architecture, each layer is based
upon the logistic regression classifier and is fully connected. The
MCDNN utilizes a two-stage novel feature learning framework.
The model is capable to automatically learn useful features
from large network traffic labeled data collection and classifies
intruders efficiently. We evaluate the effectiveness of our proposed
model on a well known publicly available challenging dataset
KDD99. The experimental results demonstrate that our proposed
framework significantly outperforms the existing models and
achieves considerable high recognition rates on KDD99.

## The MCDNN Model Architecture
![architecture](https://user-images.githubusercontent.com/35916017/72090710-8bd76600-3349-11ea-985f-c28075c59f60.png)

## Steps for running the code

### Procedure
![procedure](https://user-images.githubusercontent.com/35916017/72092201-cee70880-334c-11ea-8cd2-f9010958e122.png)

## 1. Set up the environment

    Python3
    Tensorflow
    Numpy
    Pandas
    Scikit-learn

## 2. Data set analysis and preprocessing
    Run the file named "Data_preprocesing.py"
## 3. Create and Train the model
    Run the file named "model_creation_and_training.py"
## 4. Test and Evaluate the model
    Run the file named " Model_testing_and_evaluation.py"  
    
## MCDNN Model Accuracy and Loss
![accuracy](https://user-images.githubusercontent.com/35916017/72091249-af4ee080-334a-11ea-82d8-416c711304dd.png) ![loss](https://user-images.githubusercontent.com/35916017/72091295-cbeb1880-334a-11ea-89ac-b2fd292b3237.png)





