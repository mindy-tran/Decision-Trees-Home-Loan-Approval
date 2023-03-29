import numpy as np
from my_decision_tree_classifier import MyDecisionTreeClassifier
from sklearn.model_selection import KFold

class Experiment02:

    @staticmethod
    def run():
        """
        Loads the data, sets up the machine learning model, trains the model,
        gets predictions from the model based on unseen data, assesses the
        accuracy of the model, and prints the results.
        :return: None
        """
        number_of_folds = 5

        X_data, y_data = Experiment02._load_data()
        folds = Experiment02.get_folds(k=number_of_folds, X=X_data, y=y_data)

        for fold_index, (train_X, train_y, test_X, test_y) in enumerate(folds):
            my_tree = MyDecisionTreeClassifier()

            my_tree.fit(X=train_X, y=train_y)
            predictions = my_tree.predict(X=test_X)
            accuracy = Experiment02.get_accuracy(predictions, test_y)

            print("Accuracy on fold ", fold_index, "is", accuracy)

    @staticmethod
    def get_accuracy(pred_y, true_y):
        """
        Calculates the overall percentage accuracy.
        :param pred_y: Predicted values.
        :param true_y: Ground truth values.
        :return: The accuracy, formatted as a number in [0, 1].
        0= all predictions were wrong
        1= all predictions were correct
        """
        return np.sum(pred_y==true_y)/len(true_y)

    @staticmethod
    def get_folds(k, X, y):
        """
        Partition the data into k different folds.
        :param k: The number of folds
        :param X: The samples and features which will be used for training. The data should have
        the shape:
        X =\
        [
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value],  # Sample a
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value],  # Sample b
         ...
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value]  # Sample n
        ]
        :param y: The target/response variable used for training. The data should have the shape:
        y = [target_for_sample_a, target_for_sample_b, ..., target_for_sample_n]
        :return: A list of k folds of the form [fold_1, fold_2, ..., fold_k]
        Each fold should be a tuple of the form
        fold_i = (train_X, train_y, test_X, test_y)
        """
        # prepare cross validation
        kfold = KFold(n_splits=k, shuffle=True, random_state=1)
        # enumerate splits
        result = []
        for train_index, test_index in kfold.split(X):
            train_X, test_X = X[train_index, :], X[test_index, :]
            train_y, test_y = y[train_index], y[test_index]
            # print([train_X, train_y, test_X, test_y])
            result.append([train_X, train_y, test_X, test_y])
        # print(result)
        return result

    @staticmethod
    def _load_data(filename="iris_data.csv"):
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
        X = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=(1,2))
        y = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=(0))

        return X, y


if __name__ == "__main__":
    # Run the experiment once.
    Experiment02.run()
