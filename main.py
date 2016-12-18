import createData as CD
import build_model as bm
import os
import pickle
import Users as US
from ast import literal_eval
import semanticknowledgebase as kb
import numpy as np
reload(kb)
reload(CD)
reload(bm)
reload(US)

def loadData(myUser):
    obj = None
    path = os.getcwd() + "/Data/StoredObjects/"
    fo = open(path+'IsStored','r')
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
myUser = None
myUser = US.User()
myUser.usrLogin()

print '-------------------------Content Based Filtering Started----------------------------------\n'
startCreation = loadData(myUser)
print '-----------------------Content Based Filtering Completed----------------------------------\n'

# def getModel(startCreation):
# 	buildModel = bm.Model(startCreation)
# 	return buildModel

# model = getModel(startCreation)

print '-------------------------------Building Model --------------------------------------------\n'

buildModel = bm.Model(startCreation)

print '----------------------------------------Model Built---------------------------------------\n'

X_corpus = startCreation.createData.Xcorpus

print '----------------------------Semantic Based Search Started-----------------------------------\n'
similar_movies = kb.get_similar_movies(startCreation)
input_movies = [int(m) for m in similar_movies if m not in startCreation.usr.getUsrHist()[0]]
print '---------------------------Semantic Based Search Completed----------------------------------\n'

@np.vectorize 
def selected(elem) : return elem in input_movies

selected_movies = X_corpus[selected(X_corpus[:,0])]

#print input_movies
print '---------------------------Finding Recommendations for you----------------------------------\n'
moviesIDs = buildModel.predict(selected_movies)

print '----------------------------Top 10 Recommendations for you----------------------------------\n'

idAndMovie = startCreation.createData.idAndMovie
predictions = [x[1] for x in idAndMovie if x[0] in moviesIDs]
print predictions
