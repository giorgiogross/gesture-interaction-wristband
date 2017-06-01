import numpy as np
from numpy import genfromtxt
from sklearn import tree
import sys
import os.path

sys.path.append(os.path.abspath(__file__ + "/../.."))
from input.processor.Processor import DataProcessor


class GestureScanner:
    # todo specify exact number of features later here
    FEATURES = 3

    # maximum variance on alpha and beta rotation axis (currently ignored)
    maxAlphaAtStart = 30
    maxBetaAtStart = 30
    clf = tree.DecisionTreeClassifier()

    # set up and train the machine learning instance with the raw data. This is the data which was previously recorded
    # with the recorder module
    def __init__(self, raw_data_path):
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

        self.clf.fit(train_data, train_target)

    # Creates the feature list from the data1d array. The array is expected to contain MEASUREMENT_POINTS sequential
    # measurements with x y z alpha beta gamma each.
    def _create_features(self, data1d):
        # todo calculate each feature and add it to the returned array in the right order

        # return some samples
        return [1.0, 2.0, 3.0]

    # Calculates the features of the current sensor data. The array is expected to contain MEASUREMENT_POINTS sequential
    # measurements, with x y z alpha beta gamma each.
    # Returns the predicted gesture id or -1
    def check_for_gesture(self, data1d):
        if len(data1d < DataProcessor.MEASUREMENT_POINTS * DataProcessor.MEASUREMENT_VALUES):
            return -1

        # get the first x,y and z. At least of them needs to be greater than MIN_RECORD_ACCEL specified in
        # recorder/app.py in order to check for a gesture
        if data1d[0] >= DataProcessor.MIN_RECORD_ACCEL \
                or data1d[1] >= DataProcessor.MIN_RECORD_ACCEL \
                or data1d[2] >= DataProcessor.MIN_RECORD_ACCEL:
            current_feature_values = self._create_features(data1d)
            prob = self.clf.predict([current_feature_values])
            maxIdx = np.argmax(prob)
            # print 'Calculated probability: ' + prob[maxIdx]
            if prob[maxIdx] * 100 > 80:
                return maxIdx

        return -1
