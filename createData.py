import Users as U
import HelperFunctions as hp
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

reload(U)
reload(hp)

class createDatasetsForUser(object) :
    
    def __init__(self) :
        self.usr = U.User()
        self.usr.usrLogin()
        self.usrPrf = U.UserProfile(self.usr)
        self.createData = None
        self.predictions = None #predicted values of user with movie ID
        self.yLabels = None #ylabels of id and 0/1 values
        self.XselectedCorpus = None #selected corpus from Xcorpus on which NN will be run
        self.selectedPred = None #selectedPred values from predictions which was used to build yLabels. Contains ID and predicted % values as cols
        self.X_train = None #training X dataSet for NN with ID and features columns
        self.X_test = None #testing dataSet for NN with ID and features columns
        self.y_train = None  #training y dataSet for NN with ID and 0/1
        self.y_test = None #testing y dataSet for NN with ID and 0/1
        
    def start(self,threshold) :
        self.createData = hp.createDataFramesAndIntialFeatureArray()
        #initial filepath               
        filename = "MoviesLength.csv"                  
        pathToFile = self.createData.getFilePath(filename)
        self.createData.processData(pathToFile)
        self.usrPrf.createUserProfile(self.createData.getXcorpus(),self.usr.getUsrHist())
        self.checkContentRecomm(self.createData.getXcorpus(),self.usrPrf.getUserProfile(),self.createData.getIDF())
        #discard the terms that from the corpus that are not relevant
        self.discardLowOrdPred()
        #create y labels 0/1 for selected corpus
        self.createYlabels(threshold)
        #create training and testing dataset
        self.createTrainTest()
        
    def createTrainTest(self) :
        self.X_train, self.X_test , self.y_train, self.y_test = train_test_split(self.XselectedCorpus, self.yLabels, test_size=0.20)
        # print "X_train"
        # print self.X_train
        # print self.X_train.shape
        # print "X_test"
        # print self.X_test
        # print self.X_test.shape
        # print "y_train"
        # print self.y_train
        # print self.y_train.shape
        # print "y_test"
        # print self.y_test
        # print self.y_test.shape
        
        
        self.setTrainData(self.X_train, self.y_train)
        self.setTestData(self.X_test, self.y_test)
        
        # with open("x_train.csv", "wb") as f:
        #     writer = csv.writer(f)
        #     writer.writerows(self.X_train)
        # with open("x_test.csv", "wb") as f:
        #     writer = csv.writer(f)
        #     writer.writerows(self.X_test)
        # with open("y_train.csv", "wb") as f:
        #     writer = csv.writer(f)
        #     writer.writerows(self.y_train)
        # with open("y_test.csv", "wb") as f:
        #     writer = csv.writer(f)
        #     writer.writerows(self.y_test)
        
        #print "Len of training dataSet = %d " %len(self.X_train) 
        #print "Len of testing dataset = %d " %len(self.X_test)
        n1 = self.y_train[ np.array( [x == 1 for x in self.y_train[:,1]])]       
        nn1 = self.y_test[ np.array( [x == 1 for x in self.y_test[:,1]])]       
        #print "Number of 1 labels in training set are %d " %len(n1)
        #print "Number of 1 labels in testing set are %d " %len(nn1)
                
    def checkContentRecomm(self,Xcorpus,userProfile,IDF) :      
        min_max_scaler = preprocessing.MinMaxScaler()
        temp = Xcorpus[:,1:]*userProfile.T
        #x = min_max_scaler.fit_transform(Xcorpus[:,1:].dot(userProfile.T))
        predictions = min_max_scaler.fit_transform(temp.dot(IDF.T))
        self.predictions = np.concatenate((Xcorpus[:,0].reshape(len(Xcorpus),1),predictions.reshape(len(predictions),1)),axis=1)
        #print self.predictions[1]
        #following code shows the result of content based predictions
        ind = np.argpartition(self.predictions[:,1], -5)[-5:] # indices starting from 0. Selecting top 5 matches indices
        #print ind #indexes of top 5 movies for recommendations
        #print self.predictions[ind][:,0] #print the ids of top 5 selected movies
        #print "Top 5 movies with ids " 
        #print Xcorpus[ind][:,0]
        print "Top 5 Movies "
        print self.createData.idAndMovie[ind][:,1] 
        
    def discardLowOrdPred(self,threshold=0.2) :
        X = self.createData.getXcorpus()    
        y = self.predictions
        #print y[0:10,1]
        selectedVals = np.array([ x >= threshold for x in y[:,1]])
        #print selectedVals[0:10]
        self.XselectedCorpus = X[selectedVals]
        self.selectedPred = y[selectedVals]
        #print self.XselectedCorpus[0:10]
        #print self.selectedPred[0:10]
        #print len(self.XselectedCorpus)
        #print len(self.selectedPred)
            
    def createYlabels(self,threshold = 0.7) :
        @np.vectorize
        def findVals(el) :
            if el >= threshold :
                return 1
            else :
                return 0
                
        l = len(self.selectedPred)                    
        self.yLabels = np.concatenate((self.selectedPred[:,0].reshape(l,1),findVals(self.selectedPred[:,1]).reshape(l,1)),axis=1)
        #print self.yLabels[0:4]
        ##number of values that are 1 in ylabels
        #print "Number of ylabels that are 1 = %d" %len(self.yLabels[np.array([x == 1 for x in self.yLabels[:,1]])])
    def setTrainData(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
                
    def setTestData(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test
    
    def getTrainData(self):
        return self.X_train , self.y_train
    
    def getTestData(self) :
        return self.X_test, self.y_test
        
