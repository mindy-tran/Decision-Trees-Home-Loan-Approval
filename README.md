# Decision-Trees
Implement a decision tree classifier from scratch and also create a random forest classifier with scikit-learn’s built in ensemble bagger.


Mindy Tran
Project for Machine Learning for Data Science, Winter 2023

In this project:
1. Implement a decision tree classifier from scratch with the iris dataset
* You will use the iris dataset to train and test your decision tree model.
* You will assess the accuracy of your technique using k-fold validation.
* You will create visualizations using your model.

2. Create a random forest classifier.
* Using home loan approval data from Kaggle, prepare the data for use, and train and test your random forest model on the data.
https://www.kaggle.com/datasets/rishikeshkonapure/home-loan-approval
* Use a library-provided reporting tool to produce summary statistics about the efficacy of your model.
* Use the library-provided random forest classifier on the home loan data and, using the library-provided reporting tool for insight, you will write a brief report of your findings.



*********************************************************************************
Task 0, Set up:

1. Install the required libraries for this project by running the following commands:
pip install scikit-learn
pip install scipy
pip install numpy
pip install matplotlib

2. Unzip the data.zip into your directory to get the iris_data.csv and loan_sanction_test.csv and loan_sanction_train.csv files.


*********************************************************************************
Task 1: Implement a Decision Tree Classifier

1. Run main01.py

This is the implementation of a decision tree classifier from scratch (iris dataset) with the where it trains the decision tree and provides predictions for new data.

2. Run main02.py

Assess the accuracy of your model using k-fold validation (iris dataset).
* get_accuracy(y_true, y_pred) returns a value in [0, 1] reflecting the accuracy of a list of predictions when provided with their ground truth values. (0= all predictions were wrong, 1= all predictions were correct.) 
* get_folds(k, X, y) is the method you will write to partition data into k folds to be used for training and testing. 

3. Run visualization.py

Create visualizations of your model (iris dataset).
* One image using all the data, and a single decision tree as the underlying model.
* Five images, one for each fold of 5-fold cross validation using all the data, and a single decision tree as the underlying model.


*********************************************************************************
Task 2: Implement a Random Forest Classifier

1. Run main03.py

This uses the BaggingClassifier class from scikit learn to implement your random forest
This file:

* Loads the cleaned data.
* Partitions the loaded data into two sets: training data (a randomly selected subset of your cleaned
data) and testing data (the data not selected for training). 80% training, 20% testing.
* Instantiates MyRandomForestClassifier: Trains my model on the training data. Predicts on the testing data. Save the predictions to my_random_forest_predictions. Evaluates the model with SciKit Learn Classification Report
