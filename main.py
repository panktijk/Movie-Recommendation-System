import createData as CD
import build_model as bm
import os
import pickle
import Users as US
from ast import literal_eval
reload(CD)
reload(US)


def loaddata(myUser):
    obj = None

    path = os.getcwd() + "/Data/StoredObjects/"

    fo = open(path+'isStored','r')

    v = fo.read()
    if  v =='True'  :
        with open(path+'myObj.pkl', "rb") as input:
            obj = pickle.load(input)
    else :
        obj = CD.createDatasetsForUser(myUser)
        obj.start(threshold)
        with open(path+'myObj.pkl', "wb") as output:
            pickle.dump(obj, output)
        fi = open(path + 'IsStored', 'w')
        fi.write("True")
        fi.close()

    fo.close()
    return obj

threshold = 0.5
#startCreation = CD.createDatasetsForUser()
#startCreation.start(threshold)
myUser = None
myUser = US.User()
myUser.usrLogin()

startCreation = loaddata(myUser)

buildModel = bm.Config(startCreation)
