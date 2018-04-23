# Classification of HTRU2 Data Set
This is an exploration into using TensorFlow to classify a data set.

## Data Set
The data set can be found [here](https://archive.ics.uci.edu/ml/datasets/HTRU2 "UCI Machine Learning Repository"). 
It includes 17,898 records with 1,639 real pulsar examples and 16,259 negative examples caused by interference/noise.
There are eight continuous variables and a binary class variable.

## Goals
- [x] Determine data set
- [x] Build TensorFlow from source
- [x] Classify data set
- [x] Explore alternative classification techniques
- [x] Write paper about methods tested

## Paper
The paper will be written to [NIPS standards](https://nips.cc/Conferences/2017/PaperInformation/StyleFiles), utilizing [nips_2017.sty](https://media.nips.cc/Conferences/NIPS2017/Styles/nips_2017.sty). The paper is located at:

`https://github.com/macattackftw/ResearchProject_HTRU2/blob/master/paper/paper.pdf`


## Results
Accuracy of training data is over 97%.
![Accuracy over training data](images/Accuracy_smoothed.png "Training accuracy, smoothed")


![Accuracy over training data](images/Accuracy.svg "Training accuracy")


Loss function approaches zero very quickly.
![Loss while training data](images/Loss_smoothed.png "Loss Progression")

## Build
To utilize this repository you will need the following:
- [Python3](https://www.python.org/download/releases/3.0/)
- [TensorFlow](https://www.tensorflow.org/)
- [Pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

After installing those you will need to download the [dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00372/HTRU2.zip) and place `HTRU_2.csv` in the same folder 
as `main.py`. You will then be able to run `main.py`. If you wish to see **my** results 
you can run the following from the main directory: 

`tensorboard --logdir model` 

This will will allow you to see all of the visualizations associated with this repository. 
Otherwise, if you wish to see how your model ran it should have saved under "model_X", 
where "X" is the run you wish to view. Command: 

`tensorboard --logdir model_X`.
