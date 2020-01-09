This repo contains all files required for the MCDNN implementation.

# Multi-Classifier-Deep-Neural-Network-for-Detecting-Intruder-behavior-in-Cyber-Security
**Paper Abstract**


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

## Steps for running the code

### 1. Set up the environment

    Python3
    Tensorflow
    Numpy
    Pandas
    Scikit-learn

### Data set analysis and preprocessing
    Run the file named "Data_preprocesing.py"
### Create and Train the model
    Run the file named "model_creation_and_training.py"
### Test and Evaluate the model
    Run the file named " Model_testing_and_evaluation.py"  


