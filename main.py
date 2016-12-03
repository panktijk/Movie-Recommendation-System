import createData as CD
import build_model as bm
reload(CD)

threshold = 0.5        
startCreation = CD.createDatasetsForUser()
startCreation.start(threshold)

buildModel = bm.Config(startCreation)
