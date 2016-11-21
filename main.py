import createData as CD
reload(CD)

threshold = 0.7        
startCreation = CD.createDatasetsForUser()
startCreation.start(threshold)
