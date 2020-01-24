import gensim
import numpy as np
import nltk
import string
from numpy import linalg as LA
import os
import os.path
from nltk.corpus import stopwords
import multiprocessing as mp
import datetime
from threading import Thread
#from nltk.corpus import wordnet as wn
from vectorOperations import sumVectors, getVectorModel, mergeMaxItems, mergeVectors,getListVectors
from sklearn.cluster import KMeans
from glove import *


def fillVectorsTrain(tupleStopWords, vectorsTrain, dictTrain, model, filterWordsMainTest,size):
  print("filled vectors train",datetime.datetime.now())  
  translator=str.maketrans('','',string.punctuation)
  counter=0
  cc=0
  for key in dictTrain:

    cc+=1
    idTestStringL=key.split('.')
    
    dictExercise=dictTrain[key]
    
    sentence=dictExercise['context'] #sentence.replace(" n't","n't")
    listsWords=sentence.split(dictExercise['target'])

    if(filterWordsMainTest==True):
      filteredWords=[word for word in listWords if word not in tupleStopWords]      
      tupleWords=tuple(filteredWords)
      newVectorDef=mergeVectors(translator,tupleWords,model,size)#,idTestStringL[0])
      #newVectorDef=getListVectors(tupleWords,translator,model)
      vectorsTrain[key]=newVectorDef
      
    else:
      counterOrations=0
      #if('add.v.bnc.00078359'==key):
      #  print(listsWords)
      for listWords in listsWords:
        if(counterOrations<2):      
          tupleWords=tuple(listWords.split())
          #print(listWords)
          newVectorDef=getListVectors(tupleWords,translator,model)
          counterOrations+=1
          #print(newVectorDef.shape,key)
          try:
            length,size=newVectorDef.shape

          except:
            size=newVectorDef.shape
            size=size[0]
            length=1

          if (key in vectorsTrain):
            #print("from left to right",lenght)
            if(length<70):
              #print("less 70, and existing key")
              #np.append(newVectorDef, np.zeros((size,(70-lenght-1)),dtype=np.float32), axis=1)

              if(length!=1):
                vectorsTrain[key].append(np.append(newVectorDef, np.zeros(((70-length),size),dtype=np.float32), axis=0))


              else:
                #print(np.row_stack([newVectorDef,np.zeros((size,(69)))]))
                vectorsTrain[key].append(np.row_stack([newVectorDef,np.zeros(((69),size))]))
                #print(np.row_stack([newVectorDef,np.zeros(((69),size))]),np.row_stack([newVectorDef,np.zeros(((69),size))]).shape)
                #break
            else:
              #print("error when is more 70")
              vectorsTrain[key].append(newVectorDef[0:70,:])

              
          else:
            #print("from right to left",lenght)
            #from right to left                      
              
            if(length<70):
              #print("less like 70")
              if(length!=1):
                print(newVectorDef,newVectorDef.shape)
                bufferV=np.append(newVectorDef, np.zeros(((70-length),size),dtype=np.float32), axis=0)
                #print( bufferV)
              else:
                bufferV=np.row_stack([newVectorDef,np.zeros(((69),size))])
                #print(bufferV,bufferV.shape)
              vectorsTrain[key]=[np.flip(bufferV,0)]
              #break
              
            else:
              #print("se anexa clave")              
              inverseVector=np.flip(newVectorDef,0)            
              vectorsTrain[key]=[inverseVector[0:70,:]]
            
        else:
          break
      #break
    # if (type(newVectorDef)==np.ndarray):
    #   vectorsTrain[key]=newVectorDef      
      
    counter+=1
  
  print("filled vectors train",datetime.datetime.now())

  
def fillVectorsDefinition(tupleStopWords,vectorsDefinitions, senseDict,model,filterWordsMainTest,size):
  print("filled vectors def",datetime.datetime.now())
  translator=str.maketrans('','',string.punctuation)
  
  counter=0
  for word in senseDict:
    definitions=senseDict[word]
    localVectorsDefinitions={}
    #print ("definitions word")
    #if (word=="hot%5:00:00:warm:03"):
    #  print("definitions word",word)
    for definition in definitions:
      
      instanceDefinition=definitions[definition]
      instanceDefinition=instanceDefinition.replace(" n't","n't")
      listWords=instanceDefinition.split()
#      if(definition=="hot%5:00:00:warm:03"):
#        print ("d:",instanceDefinition)
        
      if(filterWordsMainTest==True):
        filteredWordsD=[word for word in listWords if word not in tupleStopWords]          
        tupleFilteredWordsD=tuple(filteredWordsD)

        #print(filteredWordsD)
        vectorDefinition=mergeVectors(translator, tupleFilteredWordsD, model, size)
#        if(definition=="2555507"):
#          print (vectorDefinition,tupleFilteredWordsD)
          

      else:
        tupleWords=tuple(listWords)
        vectorDefinition=mergeVectors(translator, tupleWords, model, size)
          
      #vectorDefinition=np.divide(vectorDefinition,lenListWordsDefinition)
      
      localVectorsDefinitions[definition]=vectorDefinition

    
    vectorsDefinitions[word]=localVectorsDefinitions

    sleep(0.01)
  #######################################################  
  print("filled vectors def",datetime.datetime.now())



if ('dicTrain' not in locals()):
  exec(open('parseTrainFile.py').read())


if ('senseDict' not in locals()):
  exec(open('parsDic.py').read())

print("parsed files!")
    
if ('model' not in locals()):
  #model=load_glove(200)
  #model=gensim.models.KeyedVectors.load_word2vec_format("/home/arocha2/word2vec/word2vec-master/vectors.bin",binary=True)
  model=gensim.models.KeyedVectors.load_word2vec_format("~/bnc-word2vec/vectors_nnet_100.txt",binary=False)
  #model=gensim.models.KeyedVectors.load_word2vec_format("/home/arocha2/Downloads/vectors_nnet_500.txt",binary=False)
  #model=gensim.models.KeyedVectors.load_word2vec_format("/extra/arocha/representaciones vectoriales/word2vec/vectorsEn.bin",binary=False)
  #model=gensim.models.KeyedVectors.load_word2vec_format("/extra/arocha/representaciones vectoriales/word2vec/GoogleNews-vectors-negative300.bin",binary=True)

print(datetime.datetime.now())

filterWordsMainTest=False

listStopWords=stopwords.words('english')
tupleStopWords=tuple(listStopWords)

vectorsTrain={}
vectorsDefinitions={}
size=100

fillVectorsDefinition(tupleStopWords, vectorsDefinitions, senseDict, model, filterWordsMainTest, size)
fillVectorsTrain(tupleStopWords, vectorsTrain, dictTrain, model, filterWordsMainTest, size)
#t1=Thread(target=fillVectorsDefinition,args=(tupleStopWords,vectorsDefinitions, senseDict,model,filterWordsMainTest))
#t2=Thread(target=fillVectorsTrain,args=(tupleStopWords,vectorsTrain,dictTrain, model, filterWordsMainTest))

#t1.start()
#t2.start()

#t1.join()
#t2.join()

print(datetime.datetime.now())

sleep(1)
arrayTest=[]
maxShape=0
ii=0

for key in vectorsTrain:
  listFields=key.split('.')
  idSense=dictTrain[key]['answer']
  if(idSense!='U'):
    arrayTest.append((vectorsTrain[key],vectorsDefinitions[listFields[0]][idSense]))
    #print(arrayTest[0][0],type(arrayTest[0][0][1]),arrayTest[0][1])
    #ii+=1
    #if(ii==2):
    #  break
    #arrayTest.append((vectorsTrain[key].copy(),vectorsDefiabnitions[listFields[0]][idSense]))
    #if (maxShape<vectorsTrain[key].shape[1]):
    #  maxShape=vectorsTrain[key].shape[1]
      

print ("after create arrayTest")
shapeInput=size
print ("shape")
ir1=0

zd=np.zeros((len(arrayTest),70,100))
zd2=np.zeros((len(arrayTest),70,100))
zd3=np.zeros((len(arrayTest),100))
sleep(1)
print("after put zd and zd2")


for trainCase in arrayTest:
  
  try:
    zd[ir1]=trainCase[0][0]#.resize(shapeInput[0],maxShape)
    zd2[ir1]=trainCase[0][1]
    zd3[ir1]=trainCase[1]
  except:
    print("exception")#,trainCase[0],trainCase[1])
    print(type(trainCase[1]),type(trainCase[0][1]),type(trainCase[0][0]))
    print(trainCase[1].shape,trainCase[0][1].shape,trainCase[0][0].shape)
    
  ir1+=1

     

