import numpy as np
import time
from numpy import genfromtxt
# some classifiers
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
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

    train_data = np.ndarray(
        shape=(
            0,
            0
        ),
        dtype=float
    )
    train_target = []
    entries = 0

    clf = RandomForestClassifier(n_estimators=100)

    # set up and train the machine learning instance with the raw data. This is the data which was previously recorded
    # with the recorder module
    def __init__(self, raw_data_path, print_stats=False):
        measurements = genfromtxt(raw_data_path, delimiter=',', dtype=float)
        self.entries = len(measurements)
        if self.entries == 0:
            return

        lastIdx = DataProcessor.MEASUREMENT_POINTS * DataProcessor.MEASUREMENT_VALUES

        # Sort data by gesture id (at last index...)
        measurements = measurements[measurements[:,lastIdx].argsort()]

        self.train_target = measurements[:, lastIdx]
        measurements = np.delete(measurements, -1, 1)
        self.train_data = np.ndarray(
            shape=(
                len(measurements),
                GestureScanner.FEATURES
            ),
            dtype=float
        )
        for i in range(0, self.entries):
            self.train_data[i] = self._create_features(measurements[i])

        self.clf.fit(self.train_data, self.train_target)

        if print_stats:
            self._print_stats()

    def _print_stats(self):
        train = self.train_data
        target = self.train_target

        numLeft = list(target).count(0)
        numRight = list(target).count(1)
        print "Raw data contains " + repr(numRight) + " recordings for RIGHT SWIPE and " \
              + repr(numLeft) + " recordings for LEFT SWIPE"

        # match rows for left and right swipe
        diff = abs(numRight - numLeft)
        if numRight <= numLeft:
            for i in range(numLeft - diff, numLeft):
                target = np.delete(target, numLeft-diff-1, axis=0)
                train = np.delete(train, numLeft-diff-1, axis=0)
        else:
            for i in range(numLeft + numRight - diff, numLeft + numRight):
                target = np.delete(target, numLeft + numRight - diff-1, axis=0)
                train = np.delete(train, numLeft + numRight - diff-1, axis=0)
        print "Dropped redundant entries"
        print ""

        # set up our classifiers
        clfRndForest10 = RandomForestClassifier(n_estimators=10)
        clfRndForest10Name = "Random Forest with 10 estimators"
        clfRndForest100 = RandomForestClassifier(n_estimators=100)
        clfRndForest100Name = "Random Forest with 100 estimators"
        clfDecisionTree = tree.DecisionTreeClassifier()
        clfDecisionTreeName = "Decision Tree"
        clf5NN = KNeighborsClassifier(n_neighbors=5)
        clf5NNName = "5 Nearest Neighbours"
        clf11NN = KNeighborsClassifier(n_neighbors=11)
        clf11NNName = "11 Nearest Neighbours"
        clfSVM = svm.SVC()
        clfSVMName = "SVM"
        clfGausBayes = GaussianNB()
        clfGausBayesName = "Gaussian Bayes"

        # cross validation
        self._leave_one_out_validation(clfRndForest10Name, clfRndForest10, train, target)
        self._leave_one_out_validation(clfRndForest100Name, clfRndForest100, train, target)
        self._leave_one_out_validation(clfDecisionTreeName, clfDecisionTree, train, target)
        self._leave_one_out_validation(clf5NNName, clf5NN, train, target)
        self._leave_one_out_validation(clf11NNName, clf11NN, train, target)
        self._leave_one_out_validation(clfSVMName, clfSVM, train, target)
        self._leave_one_out_validation(clfGausBayesName, clfGausBayes, train, target)

        self._nfold_cross_validation(10, clf5NNName, clf5NN, train, target)

    def _leave_one_out_validation(self, clfName, clf, train, target):
        print "- - - - - - - -              Leave one out results:              - - - - - - - -"
        right_classified_samples = 0.0
        wrong_classified_samples = 0.0
        train_samples = (target.size * 1.0) - 1
        test_samples = 1.0
        samples = target.size
        time_approx = 0

        print ">" + clfName + ":"

        for n in range(0, samples):
            # extract one element
            m_test_train = [train[n]]
            m_test_target = [target[n]]
            m_train = np.delete(train, n, axis=0)
            m_target = np.delete(target, n, axis=0)

            # train classifier with remaining elements
            clf.fit(m_train, m_target)

            # test
            ct = time.time()
            predictions = clf.predict(m_test_train)
            time_approx += time.time() - ct

            for i in range(0, predictions.size):
                if predictions[i] == m_test_target[i]:
                    right_classified_samples += 1
                else:
                    wrong_classified_samples += 1
            self._update_progress(round(n / (samples * 1.0), 2) + 0.01)

        # compute results
        time_approx = int(round(time_approx * 1000000.0 / samples))
        precision = 0.0
        accuracy = 0.0
        if right_classified_samples != 0:
            precision = round((right_classified_samples + wrong_classified_samples) / right_classified_samples, 3)
            accuracy = round(right_classified_samples / (right_classified_samples + wrong_classified_samples), 3)

        # print stats
        print " PRECISION=" + repr(precision)+"    ACCURACY=" + repr(accuracy) \
              + "    RECALL=" + repr(right_classified_samples) + "    ~COMP.TIME=" + repr(time_approx) + "mis"
        print "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
        print "\n\n"

    def _nfold_cross_validation(self, n, clfName, clf, train, target):
        print "- - - - - - - -              " + repr(n) + "fold cross validation:              - - - - - - - -"
        right_classified_samples = 0.0
        wrong_classified_samples = 0.0
        train_samples = (target.size * 1.0) - 1
        test_samples = 1.0
        samples = target.size
        time_approx = 0

        if n > samples:
            print "Please choose a smaller value"
            return;
        print ">" + clfName + ":"

        # modify train and target arrays properly
        ratio = samples / (n * 1.0)
        if not ratio.is_integer():
            ratio = (int)(round(ratio))
            validSamples = ratio * n
            dropNum = samples - validSamples
            for i in range(validSamples, validSamples + dropNum):
                train = np.delete(train, validSamples)
                target = np.delete(target, validSamples)
        samples = target.size
        print target.size

        for n in range(0, samples):
            # extract one element
            m_test_train = [train[n]]
            m_test_target = [target[n]]
            m_train = np.delete(train, n, axis=0)
            m_target = np.delete(target, n, axis=0)

            # train classifier with remaining elements
            clf.fit(m_train, m_target)

            # test
            ct = time.time()
            predictions = clf.predict(m_test_train)
            time_approx += time.time() - ct

            for i in range(0, predictions.size):
                if predictions[i] == m_test_target[i]:
                    right_classified_samples += 1
                else:
                    wrong_classified_samples += 1
            self._update_progress(round(n / (samples * 1.0), 2) + 0.01)

        # compute results
        time_approx = int(round(time_approx * 1000000.0 / samples))
        precision = 0.0
        accuracy = 0.0
        if right_classified_samples != 0:
            precision = round((right_classified_samples + wrong_classified_samples) / right_classified_samples, 3)
            accuracy = round(right_classified_samples / (right_classified_samples + wrong_classified_samples), 3)

        # print stats
        print " PRECISION=" + repr(precision)+"    ACCURACY=" + repr(accuracy) \
              + "    RECALL=" + repr(right_classified_samples) + "    ~COMP.TIME=" + repr(time_approx) + "mis"

    #X_train, X_test, y_train, y_test = train_test_split(self.train_data, self.train_target, test_size= .5)
    #print accuracy_score(y_test, predictions)

    def _update_progress(self, p):
        pr = int(round(p*100))
        sys.stdout.write("\r%d%%" % pr)
        if pr >= 100:
            sys.stdout.write("\r")
        sys.stdout.flush()

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
