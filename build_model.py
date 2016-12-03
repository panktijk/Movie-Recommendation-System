import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import createData as CD

class Config(object):   

    def __init__(self, object) :
        self.X_train, self.y_train = object.getTrainData()
        self.X_test, self.y_test = object.getTestData()
        self.yTrain = np.array(self.y_train[:,1], dtype = int)
        self.yTest = np.array(self.y_test[:,1], dtype = int)
        self.classify(self.X_train, self.yTrain, self.X_test, self.yTest)

    def classify(self, X_train, yTrain, X_test, yTest):
        # clf = linear_model.LogisticRegressionCV(max_iter = 20000, solver = 'newton-cg')
        # w = clf.fit(X_train, yTrain)
        # print w
        # x = clf.predict(X_test)
        # print x
        # print "clf.decision_function(X_test) = ", clf.decision_function(X_test)
        # print clf.score(X_test, yTest)
        # print

        clf = MLPClassifier(solver = 'lbfgs', hidden_layer_sizes=(5, ), activation = 'logistic', max_iter = 200)
        print clf.fit(X_train, yTrain)
        print X_train.shape
        print yTrain.shape
        print yTest
        print "Prediction of x_test"
        print clf.predict(X_test)
        print clf.score(X_test, yTest)
        print clf.predict_proba(X_test)



