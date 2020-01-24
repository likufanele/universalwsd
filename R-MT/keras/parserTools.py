import numpy as np
import re
import string
from vectorOperations import sumVectors, getVectorModel, mergeVectors, distanciaE, getListVectors 

def splitOrationsAndPad(key,listsWords,vectorsSet,translator,modelV,size):
    counterOrations=0
    #print(len(listsWords))
    sizeLen=15
    translator=str.maketrans('','',string.punctuation)
    for listWords in listsWords:
        #print(counterOrations)
        if(counterOrations<2):
            filtredListWords=re.sub('[,.;:]', '', listWords)            
            tupleWords=tuple(filtredListWords.split())
            newVectorDef=getListVectors(tupleWords,translator,modelV,size)
            counterOrations+=1
            
            try:
                size,length=newVectorDef.shape

            except:
              size=newVectorDef.shape
              size=size[0]
              length=1

            if (key in vectorsSet):
                #print("existe clave")
                if(length<sizeLen):
                    #print("less 70, and existing key")                    
                    if(length!=1):
                        vectorsSet[key].append(np.append(newVectorDef, np.zeros((size,(sizeLen-length)),dtype=np.float32), axis=1))

                    else:
                        #print(np.column_stack([newVectorDef,np.zeros((size,(69)))]))
                        vectorsSet[key].append(np.column_stack([newVectorDef,np.zeros((size,(49)))])) #sizelen-1
                
                else:
                    #print("error when is more 70")
                    vectorsSet[key].append(newVectorDef[:,0:50])
                    
            else:
                #print("from right to left",length)
                #from right to left                      
              
                if(length<sizeLen):
                    #print("less like 70")
                    if(length!=1):
                        bufferV=np.append(newVectorDef, np.zeros((size,(sizeLen-length)),dtype=np.float32), axis=1)
                    else:
                        bufferV=np.column_stack([newVectorDef,np.zeros((size,(49)))]) #sizelen-1
                  
                    vectorsSet[key]=[np.flip(bufferV,1)]
              
                else:
                    #print("se anexa clave",newVectorDef.shape)              
                    inverseVector=np.flip(newVectorDef,1)            
                    vectorsSet[key]=[inverseVector[:,-50:]]
              
        else:
          break


def splitOrationsAndPadB(key,listsWords,vectorsSet,sizeLen,modelV,size):
    counterOrations=0
    #sizeLen=15
    translator=str.maketrans('','',string.punctuation)
    for listWords in listsWords:
        #print(counterOrations)
        if(counterOrations<2):
            filtredListWords=re.sub('[,.;:]', '', listWords)            
            tupleWords=tuple(filtredListWords.split())
            #3print(tupleWords)
            
            newVectorDef=getListVectors(tupleWords,translator,modelV,size) ##remember change column for row
            counterOrations+=1

            try:
              length,size=newVectorDef.shape

            except:
              sizeD=newVectorDef.shape
              size=sizeD[0]
              length=1

            if (key in vectorsSet):
                #print("left side",tupleWords )
                if(length<sizeLen):
                    #print("less 70, and existing key")                    
                    
                    if(length!=1):
                        
                        newVectorDef=np.append(newVectorDef, np.zeros(((sizeLen-length),size),dtype=np.float32), axis=0)
                        newVectorDef=np.flipud(newVectorDef)
                        vectorsSet[key].append(newVectorDef)
                                                                        
                    else:
                        #print(np.column_stack([newVectorDef,np.zeros((size,(69)))]))
                        vectorsSet[key].append(np.row_stack([newVectorDef,np.zeros(((sizeLen-1),size))]))
                
                else:
                    
                    newVectorDef=newVectorDef[0:sizeLen,:]
                    
                    newVectorDef=np.flipud(newVectorDef)
                    vectorsSet[key].append(newVectorDef)

                #import pdb; pdb.set_trace()
                
            else:
                #print("right side",tupleWords )
                if(length<sizeLen):
                    #print("less like 5")
                    #testS.append(list(tupleWords))
                    
                    if(length!=1):
                        #print("right side",tupleWords)
                        #print(list(tupleWords[-sizeLen:]))
                        #import pdb; pdb.set_trace()
                        bufferV=np.zeros(((sizeLen-length),size),dtype=np.float32)               
                        bufferV=np.append(bufferV,newVectorDef, axis=0)

                        #for i in range(0,(sizeLen-length)):
                        #    testS.append(' ')
                        
                    else:
                        bufferV=np.row_stack([np.zeros(((sizeLen-1),size)),newVectorDef])
                        #for i in range(0,(sizeLen-1)):
                        #    testS.append(' ')

                    #print("rightSide",tupleWords)
                    #print(tupleWords[-sizeLen:])
                    #import pdb; pdb.set_trace()
                    vectorsSet[key]=[bufferV[-sizeLen:]]  #np.flip(bufferV,0)]
                    
                else:
                    #print("se anexa clave")              
                    #inverseVector=bufferV[-sizeLen:] #np.flip(newVectorDef,0)            
                    vectorsSet[key]=[newVectorDef[-sizeLen:]]#[inverseVector[-sizeLen:,:]] 
                    #print(tupleWords,tupleWords[-sizeLen:])
                    #import pdb; pdb.set_trace()
            
        else:
          #import pdb; pdb.set_trace()  
          break
      
    partsKey=key.split('.')
    vectorsSet[key].append(getVectorModel(partsKey[0],modelV))
    #import pdb; pdb.set_trace()
    

def fillVectorsDefinition(vectorsDefinitions, senseDict,modelVectors,size):
  print("filled vectors def",datetime.datetime.now())
  translator=str.maketrans('','',string.punctuation)
  
  counter=0
  for word in senseDict:
    definitions=senseDict[word]
    localVectorsDefinitions={}

    for definition in definitions:
      
      instanceDefinition=definitions[definition]
      instanceDefinition=instanceDefinition.replace(" n't","n't")
      listWords=instanceDefinition.split()
              
      tupleWords=tuple(listWords)
      vectorDefinition=mergeVectors(translator, tupleWords, modelVectors, size)          
      
      localVectorsDefinitions[definition]=vectorDefinition

    
    vectorsDefinitions[word]=localVectorsDefinitions

    sleep(0.01)
  #######################################################  
  print("filled vectors def",datetime.datetime.now())
