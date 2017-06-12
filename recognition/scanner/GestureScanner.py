import numpy as np
from numpy import genfromtxt
# some classifiers
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import sys

# data processing constants are needed in this file
import os.path
sys.path.append(os.path.abspath(__file__ + "/../.."))
from input.processor.Processor import DataProcessor
# Tree diagram generation
from sklearn.externals.six import StringIO
import pydot
# Classifier testing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


class GestureScanner:
    # todo specify exact number of features later here
    FEATURES = 18
    POINTS_FOR_FEATURE = 3

    # maximum variance on alpha and beta rotation axis (currently ignored)
    maxAlphaAtStart = 30
    maxBetaAtStart = 30
    clf = RandomForestClassifier(n_estimators=100)

    # set up and train the machine learning instance with the raw data. This is the data which was previously recorded
    # with the recorder module
    def __init__(self, raw_data_path, print_accuracy=False):
        measurements = genfromtxt(raw_data_path, delimiter=',', dtype=float)
        entries = len(measurements)
        if entries == 0:
            return
        lastIdx = DataProcessor.MEASUREMENT_POINTS * DataProcessor.MEASUREMENT_VALUES

        train_target = measurements[:, lastIdx]
        measurements = np.delete(measurements, -1, 1)
        train_data = np.ndarray(
            shape=(
                len(measurements),
                GestureScanner.FEATURES
            ),
            dtype=float
        )

        for i in range(0, entries):
            train_data[i] = self._create_features(measurements[i])

        # split array in half to eventually test the accuracy of our classifier
        X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size= .5)
        if print_accuracy:
            train_data = X_train
            train_target = y_train

        self.clf.fit(train_data, train_target)

        if print_accuracy:
            predictions = self.clf.predict(X_test)
            print 'Accuracy of the classifier:'
            print accuracy_score(y_test, predictions)


    # Create a pdf file showing how the tree classifier looks like. Only call this with a tree.DecisionTreeClassifier!
    def _create_tree_diagram(self, clf):
        dot_data = StringIO()
        tree.export_graphviz(self.clf,
                             out_file=dot_data,
                             feature_names=[
                                 'min x',
                                 'max x',
                                 'min y',
                                 'max y',
                                 'min z',
                                 'max z',
                                 'acc alpha +',
                                 'acc alpha -',
                                 'acc beta +',
                                 'acc beta -',
                                 'acc gamma +',
                                 'acc gamma -',
                             ],
                             class_names=[
                                 'swipe left',
                                 'swipe right'
                             ],
                             filled=True, rounded=True,
                             impurity=False)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        graph[0].write_pdf("tree.pdf")

    # Creates the feature list from the data1d array. The array is expected to contain MEASUREMENT_POINTS sequential
    # measurements with x y z alpha beta gamma each.
    def _create_features(self, data1d):
        # 1. min/max per axis
        x_values = y_values = z_values = alpha_values = beta_values = gamma_values = np.array([], dtype=float)
        alpha_acc = alpha_acc_neg = beta_acc = beta_acc_neg = gamma_acc = gamma_acc_neg = 0
        for i in range(0, GestureScanner.POINTS_FOR_FEATURE):
            x_values = np.insert(x_values, 0, data1d[DataProcessor.MEASUREMENT_VALUES * i])
            y_values = np.insert(y_values, 0, data1d[DataProcessor.MEASUREMENT_VALUES * i + 1])
            z_values = np.insert(z_values, 0, data1d[DataProcessor.MEASUREMENT_VALUES * i + 2])
            alpha_values = np.insert(alpha_values, 0, data1d[DataProcessor.MEASUREMENT_VALUES * i + 3])
            beta_values = np.insert(beta_values, 0, data1d[DataProcessor.MEASUREMENT_VALUES * i + 4])
            gamma_values = np.insert(gamma_values, 0, data1d[DataProcessor.MEASUREMENT_VALUES * i + 5])
            alpha_acc += max(0, data1d[DataProcessor.MEASUREMENT_VALUES * i + 3])
            alpha_acc_neg -= min(0, data1d[DataProcessor.MEASUREMENT_VALUES * i + 3])
            beta_acc += max(0, data1d[DataProcessor.MEASUREMENT_VALUES * i + 4])
            beta_acc_neg -= min(0, data1d[DataProcessor.MEASUREMENT_VALUES * i + 4])
            gamma_acc += max(0, data1d[DataProcessor.MEASUREMENT_VALUES * i + 5])
            gamma_acc_neg -= min(0, data1d[DataProcessor.MEASUREMENT_VALUES * i + 5])

        x_min = np.argmin(x_values)
        x_max = np.argmax(x_values)
        y_min = np.argmin(y_values)
        y_max = np.argmax(y_values)
        z_min = np.argmin(z_values)
        z_max = np.argmax(z_values)
        alpha_min = np.argmin(alpha_values)
        alpha_max = np.argmax(alpha_values)
        beta_min = np.argmin(beta_values)
        beta_max = np.argmax(beta_values)
        gamma_min = np.argmin(gamma_values)
        gamma_max = np.argmax(gamma_values)

        # return the features
        return np.array([
            x_values[x_min] if (x_min < x_max) else x_values[x_max],
            x_values[x_min] if (x_min > x_max) else x_values[x_max],
            y_values[y_min] if (y_min < y_max) else y_values[y_max],
            y_values[y_min] if (y_min > y_max) else y_values[y_max],
            z_values[z_min] if (z_min < z_max) else z_values[z_max],
            z_values[z_min] if (z_min > z_max) else z_values[z_max],
            alpha_values[alpha_min] if (alpha_min < alpha_max) else alpha_values[alpha_max],
            alpha_values[alpha_min] if (alpha_min > alpha_max) else alpha_values[alpha_max],
            beta_values[beta_min] if (beta_min < beta_max) else beta_values[beta_max],
            beta_values[beta_min] if (beta_min > beta_max) else beta_values[beta_max],
            gamma_values[gamma_min] if (gamma_min < gamma_max) else gamma_values[gamma_max],
            gamma_values[gamma_min] if (gamma_min > gamma_max) else gamma_values[gamma_max],
            alpha_acc,
            alpha_acc_neg,
            beta_acc,
            beta_acc_neg,
            gamma_acc,
            gamma_acc_neg
        ], dtype=float)

    # Calculates the features of the current sensor data. The array is expected to contain MEASUREMENT_POINTS sequential
    # measurements, with x y z alpha beta gamma each.
    # Returns the predicted gesture id or -1
    def check_for_gesture(self, data1d):
        if len(data1d) < DataProcessor.MEASUREMENT_POINTS * DataProcessor.MEASUREMENT_VALUES:
            return -1

        # check if acceleration vector indicates the start of a gesture just like we did when recording the gesture
        accelVector = data1d[0:3:1]
        if DataProcessor.is_gesture_start(accelVector):
            current_feature_values = self._create_features(data1d)
            prob = self.clf.predict_proba([current_feature_values])[0]
            print 'Probability:'
            print prob
            maxIdx = np.argmax(prob)
            if prob[maxIdx] * 100 > 80:
                return maxIdx

        return -1

    def gesture_proba(self, data1d):
        if len(data1d) < DataProcessor.MEASUREMENT_POINTS * DataProcessor.MEASUREMENT_VALUES:
            return -1

        # check if acceleration vector indicates the start of a gesture just like we did when recording the gesture
        accelVector = data1d[0:3:1]
        if DataProcessor.is_gesture_start(accelVector):
            current_feature_values = self._create_features(data1d)
            return self.clf.predict_proba([current_feature_values])[0]
