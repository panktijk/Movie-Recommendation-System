import pandas as pd
import numpy as np

class User(object) :
    def __init__(self) :
        self.name = None
        self.user = None
        self.userHist = None

    def getUsrHist(self):
        return self.userHist
            
    def usrLogin(self) :
        
        while True :
            userName = raw_input("Enter Username: ")
            passw = raw_input("Enter Password: ")  
        
            if userName.lower().strip() == "ajay" :
                self.user = userName
                self.name = userName.upper()
                if passw.strip() == "11" :
                    print "\n-------------------Welcome %s----------------------\n\n"  % (self.name)
                    self.loadUserProfile()
                    break
                else :
                    print "Invalid Password"
            else :
                print "Invalid User"
            print 'Try Again!!!' 

    def loadUserProfile(self) :
        path = "Data/%s_hist.csv" %self.name.lower()
        self.userHist = pd.read_csv(path,names=["MovieID","LD"]).as_matrix()
        #print self.userHist

class UserProfile(User) :
    
    def __init__(self,user) :
        super(UserProfile,self).__init__()
        self.name = user.name
        #self.userHist = user.userHist
        self.profile = None
    
    def getUserProfile(self) :
        return self.profile
    
    def createUserProfile(self,Xcorpus,userHist) :
        @np.vectorize 
        def selected(elem) : return elem in userHist[:,0]
        XforUsers = Xcorpus[selected(Xcorpus[:,0])]
        #print XforUsers
        userProfile = userHist[:,1].T.dot(XforUsers[:,1:])
        self.profile = userProfile
        #print userProfile.shape
        #print userProfile
        #print Xcorpus[0,1:]
        
        
