import matplotlib.pyplot as plt
from my_decision_tree_classifier import MyDecisionTreeClassifier
import numpy as np
from sklearn.model_selection import KFold

# One image using all the data, and a single decision tree as the underlying model
    # makes a prediction for ∼ 10,000 equally and rectangularly spaced points

# Five images, one for each fold of 5-fold cross validation using all the data, and a single decision
# tree as the underlying model.


class Visualization:

    @staticmethod
    def run():
        """
        Loads the data, sets up the machine learning model, trains the model,
        gets predictions from the model based on unseen data, assesses the
        accuracy of the model, and prints the results.
        :return: None
        """
        train_X, train_y, test_X, test_y = Visualization.load_data()
        my_tree = MyDecisionTreeClassifier()

        # fit with 100% of data
        my_tree.fit(X=train_X, y=train_y)

        # 10,000 equally and rectangularly spaced points
        test_X, test_y = Visualization.evenly_spaced_data(train_X, train_y)
        # makes a prediction for ∼ 10,000 equally and rectangularly spaced points
        predictions = my_tree.predict(X=test_X)

        # Plot in 2D which class our model predicts for a given pair of (sepal area, petal area)
        Visualization.vis(test_X, predictions)
        # print(np.shape(test_X), np.shape(predictions))


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

        # use 100% of data for training
        test_X, test_y = [], []

        train_X = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=(1, 2))
        train_y = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=(0))

        return train_X, train_y, test_X, test_y

    @staticmethod
    def evenly_spaced_data(X, y):
        '''
        Select 10k evenly spaced points
        1. get a list of evenly spaced indices
        2. index back into arr to get the corresponding values

        :param X: np.array, features
        :param y: np.array or list, target
        :return: 10k evenly spaced points
        '''
        idx = np.round(np.linspace(0, len(X) - 1, 10_000)).astype(int)
        # return points to plot
        return X[idx], y[idx]

    @staticmethod
    def vis(test_X, predictions):
        '''
        Plot in 2D which class our model predicts for a given pair of (sepal area, petal area).

        One image using all the data, and a single decision tree as the underlying model
        makes a prediction for ∼ 10,000 equally and rectangularly spaced points

        :param predictions: array of predicted y
        :return: plot, classes our model predicts for a given pair of (sepal area, petal area).
        '''
        x_val = [x[0] for x in test_X]
        y_val = [x[1] for x in test_X]
        colors = 'cool'

        plt.scatter(x_val, y_val, c=predictions)
        plt.xlabel('Sepal Area')
        plt.ylabel('Petal Area')
        plt.ylim(-5, 20)
        plt.xlim(5, 35)
        plt.show()


class Visualization2:

    @staticmethod
    def run():
        """
        Loads the data, sets up the machine learning model, trains the model,
        gets predictions from the model based on unseen data, assesses the
        accuracy of the model, and prints the results.
        :return: None
        """
        number_of_folds = 5

        X_data, y_data = Visualization2._load_data()
        folds = Visualization2.get_folds(k=number_of_folds, X=X_data, y=y_data)

        for fold_index, (train_X, train_y, test_X, test_y) in enumerate(folds):
            my_tree = MyDecisionTreeClassifier()

            my_tree.fit(X=train_X, y=train_y)
            predictions = my_tree.predict(X=test_X)

            Visualization2.vis(test_X, predictions)


    @staticmethod
    def vis(test_X, predictions):
        '''
        Plot in 2D which class our model predicts for a given pair of (sepal area, petal area).

        One image using all the data, and a single decision tree as the underlying model
        makes a prediction for ∼ 10,000 equally and rectangularly spaced points

        :param predictions: array of predicted y
        :return: plot, classes our model predicts for a given pair of (sepal area, petal area).
        '''
        x_val = [x[0] for x in test_X]
        y_val = [x[1] for x in test_X]
        colors = 'cool'

        plt.scatter(x_val, y_val, c=predictions)
        plt.xlabel('Sepal Area')
        plt.ylabel('Petal Area')
        plt.ylim(-5, 20)
        plt.xlim(5, 35)
        plt.show()


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
    # Common bugs:
    # (1) If the output is identical each time, it means there's an
    # error with your randomized sample selection for training vs testing.
    # (2) If the accuracy is low then there is either a flaw in your model or
    # the y values are not correctly associated with their corresponding x samples.

    # One image using all the data, and a single decision tree as the underlying model
    # Visualization.run()
    # Five images, one for each fold of 5-fold cross validation using all the data, and a single decision
        # tree as the underlying model.
    Visualization2.run()

