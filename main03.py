import numpy as np
from my_random_forest_classifier import MyRandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# 1. Loads your cleaned data.
# 2. Partitions the loaded data into two sets: training data (a randomly selected subset of your cleaned
    # data) and testing data (the data not selected for training). You may choose the exact train/test
    # ratio, but it should be fairly high. E.g., 80% training, 20% testing; 90% training, 10% testing.
# 3. Instantiates your MyRandomForestClassifier.
    # • Trains your model on the training data.
    # • Predicts on the testing data. Save the my_random_forest_predictions to a variable, such as my_random_forest_my_random_forest_predictions.



class Experiment03:

    @staticmethod
    def run():
        """
        Loads the data, sets up the machine learning model, trains the model,
        gets my_random_forest_predictions from the model based on unseen data, assesses the
        accuracy of the model, and prints the results.
        :return: my_random_forest_predictions
        """
        print("Loading Data")
        train_X, train_y, test_X, test_y = Experiment03._load_data()
        my_tree = MyRandomForestClassifier()

        print("Training Model...")
        my_tree.fit(X=train_X, y=train_y)
        print("Training complete!")
        print()
        print("Evaluating Model")
        my_random_forest_predictions = my_tree.predict(X=test_X)
        y_train_pred = my_tree.predict(X=train_X)

        print("My decision tree predicted:")
        print(my_random_forest_predictions)
        print("The true values were actually:")
        print(test_y)
        print("The accuracy:")
        print(Experiment03._get_accuracy(my_random_forest_predictions, test_y))
        print()

        print("-- SciKit Learn Classification Report: Training Data")
        print(metrics.classification_report(train_y, y_train_pred))

        print("-- SciKit Learn Classification Report: Training Data")
        print(metrics.classification_report(test_y, my_random_forest_predictions))


    @staticmethod
    def _get_accuracy(pred_y, true_y):
        """
        Calculates the overall percentage accuracy.
        :param pred_y: Predicted values.
        :param true_y: Ground truth values.
        :return: The accuracy, formatted as a number in [0, 1].
        0= all my_random_forest_predictions were wrong
        1= all my_random_forest_predictions were correct
        """
        if len(pred_y) != len(true_y):
            raise Exception("Different number of prediction-values than truth-values.")

        number_of_agreements = 0
        number_of_pairs = len(true_y)  # Or, equivalently, len(pred_y)

        for individual_prediction_value, individual_truth_value in zip(pred_y, true_y):
            if individual_prediction_value == individual_truth_value:
                number_of_agreements += 1
        accuracy = number_of_agreements / number_of_pairs
        return accuracy


    @staticmethod
    def _load_data(filename="cleaned_data.csv"):
        """
        Load the data, separating it into a list of samples and their corresponding outputs
        :param filename: The location of the data to load from file.
        :return: X, y; each as an iterable object(like a list or a numpy array).
        The data should have
        the shape:
        X =\
        [
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value],  # Sample a
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value],  # Sample b
         ...
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value]  # Sample n
        ]

        y = [target_for_sample_a, target_for_sample_b, ..., target_for_sample_n]
        """
        # Modify anything in this method, but keep the return line the same.
        # You may also import any needed library (like numpy)
        x_data = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=np.arange(0, 12))
        y_data = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=12)

        train_X, test_X, train_y, test_y = train_test_split(
            x_data, y_data, test_size=0.2)

        return train_X, train_y, test_X, test_y


if __name__ == "__main__":
    # Run the experiment once.
    pred = Experiment03.run()
