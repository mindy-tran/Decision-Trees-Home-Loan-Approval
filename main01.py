from my_decision_tree_classifier import MyDecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split

class Experiment01:

    @staticmethod
    def run():
        """
        Loads the data, sets up the machine learning model, trains the model,
        gets predictions from the model based on unseen data, assesses the
        accuracy of the model, and prints the results.
        :return: None
        """
        train_X, train_y, test_X, test_y = Experiment01.load_data()
        my_tree = MyDecisionTreeClassifier()

        my_tree.fit(X=train_X, y=train_y)
        predictions = my_tree.predict(X=test_X)

        print("My decision tree predicted:")
        print(predictions)
        print("The true values were actually:")
        print(test_y)
        print("The accuracy:")
        print(Experiment01._get_accuracy(predictions, test_y))
        print()

    @staticmethod
    def load_data(filename="iris_data.csv"):
        """
        Load the data and partition it into testing and training data.
        :param filename: The location of the data to load from file.
        :return: train_X, train_y, test_X, test_y; each as an iterable object
        (like a list or a numpy array).
        """

        # Modify anything in this method, but keep the return line the same.
        # You may also import any needed library (like numpy)
        x_data = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=(1,2))
        y_data = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=(0))

        train_X, test_X, train_y, test_y = train_test_split(
            x_data, y_data, test_size = 0.2)

        return train_X, train_y, test_X, test_y

    @staticmethod
    def _get_accuracy(pred_y, true_y):
        """
        Calculates the overall percentage accuracy.
        :param pred_y: Predicted values.
        :param true_y: Ground truth values.
        :return: The accuracy, formatted as a number in [0, 1].
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


if __name__ == "__main__":
    # Run the experiment 10 times.
    # Common bugs:
    # (1) If the output is identical each time, it means there's an
    # error with your randomized sample selection for training vs testing.
    # (2) If the accuracy is low then there is either a flaw in your model or
    # the y values are not correctly associated with their corresponding x samples.
    for _ in range(10):
        Experiment01.run()
