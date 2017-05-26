import numpy as np
from numpy import genfromtxt
from sklearn import tree


class GestureScanner:
    n = 0
    clf = tree.DecisionTreeClassifier()

    # set up and train the machine learning instance with the raw data. This is the data which was previously recorded
    # with the recorder module
    def __init__(self, raw_data_path):
        raw_data = genfromtxt(raw_data_path, delimiter=',', dtype=float)

        # todo...
        n = len(raw_data)

        print raw_data
        # self._create_features() for each line of data; collect that features in an array which will be used to train the ML instance

    # Creates the feature list from the data1d array. The array is expected to contain n sequential measurements,
    # with x y z alpha beta gamma each.
    def _create_features(self, data1d):
        print data1d
        return

    # Calculates the features of the current sensor data. The array is expected to contain n sequential measurements,
    # with x y z alpha beta gamma each.
    # Returns the predicted gesture id or -1
    def check_for_gesture(self, data1d):
        # todo manage start / stop listening based on angles

        if len(data1d < self.n):
            return -1

        current_feature_values = self._create_features(data1d)
        # todo predict gesture and pay attention to probability

        return -1
