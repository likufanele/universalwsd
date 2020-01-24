import numpy as np

def mergeMaxItems(npArray1,npArray2):
    return np.maximum(npArray1,npArray2)
    
def mergeVectors(listS,model):
  modelS=np.array([])
  for i in listS:
    if(i in model.vocab):
     if(modelS.size!=0):
       modelS=mergeMaxItems(modelS,model[i])
     else:
       modelS=model[i]
  return modelS
