import gensim
import numpy as np
import nltk
import string
from numpy import linalg as LA
import os
import os.path
import random
from nltk.corpus import stopwords
import datetime
from vectorOperations import sumVectors, getVectorModel, mergeMaxItems, mergeVectors,getListVectors
from glove import *
from parserTools import *
import re
from sklearn.decomposition import PCA
#from gensim.test import utils #datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sequence import *
from configparser import ConfigParser

def fillVectorsTrain(tupleStopWords, vectorsTrain, dictTrain, modelVectors, filterWordsMainTest,size):
  print("filled vectors train",datetime.datetime.now())  
  translator=str.maketrans('','',string.punctuation)
  counter=0

  lenSize=2
  for key in sorted(dictTrain):

    idTestStringL=key.split('.')
    
    dictExercise=dictTrain[key]
    #import pdb; pdb.set_trace()
    listsWords=dictExercise['contextTag'].split(dictExercise['head']) #sentence.replace(" n't","n't")
    
    listsWords[0]= re.sub('<[^>]*>', '',listsWords[0]).rstrip()
    listsWords[1]= re.sub('<[^>]*>', '',listsWords[1]).rstrip()
    
    if(len(listsWords)>2):
      #import pdb; pdb.set_trace()
      listsWords[2]= re.sub('<[^>]*>', '',listsWords[2]).rstrip()
      complementTarget = re.sub('<[^>]*>', '',dictExercise['head']).rstrip()
      listsWords[1]=(listsWords[1]+' '+complementTarget+' '+listsWords[2])
      if(len(listsWords)>3):
        listsWords[3]= re.sub('<[^>]*>', '',listsWords[3]).rstrip()
        listsWords[1]=(listsWords[1]+' '+complementTarget+' '+listsWords[3])
        #import pdb; pdb.set_trace()

        
    if(filterWordsMainTest==True):
      filteredWords=[word for word in listWords if word not in tupleStopWords]      
      tupleWords=tuple(filteredWords)
      newVectorDef=mergeVectors(translator,tupleWords,modelVectors,size)#,idTestStringL[0])
      #newVectorDef=getListVectors(tupleWords,translator,model)
      vectorsTrain[key]=newVectorDef
      
    else:      
      splitOrationsAndPadB(key,listsWords,vectorsTrain, lenSize, modelVectors,size)
      #import pdb; pdb.set_trace()x
      #print(key)
      #break  
      
    counter+=1
  
  print("filled vectors train",datetime.datetime.now())
  

  
def fillVectorsDefinition(tupleStopWords,vectorsDefinitions, senseDict,modelVectors,filterWordsMainTest,size):

  print("filled vectors def",datetime.datetime.now())
  translator=str.maketrans('','',string.punctuation)
  counter=0

  maxLengthDef=0
  averageLengthDef=0
  counterDef=0
  for word in senseDict:
    definitions=senseDict[word]
    for definition in definitions:
      instanceDefinition=definitions[definition]
      instanceDefinition=instanceDefinition.replace(" n't","n't")
      listWords=instanceDefinition.split()
      if(len(listWords)>maxLengthDef):
        maxLengthDef=len(listWords)
      averageLengthDef+=len(listWords)
      counterDef+=1
  print("maxLengthDef:",maxLengthDef)
  print("averageLengthDef:",averageLengthDef/counterDef)
  ###########################################3
  for word in senseDict:
    definitions=senseDict[word]
    localVectorsDefinitions={}
    for definition in definitions:      
      instanceDefinition=definitions[definition]
      instanceDefinition=instanceDefinition.replace(" n't","n't")
      listWords=instanceDefinition.split()
        
      tupleWords=tuple(listWords)
      #import pdb; pdb.set_trace()
      vectorDefinition1=formList(instanceDefinition,translator,modelVectors,10)
      vectorDefinition2=formList(instanceDefinition,translator,modelVectors,10,True)
      vectorDefinition=mergeVectors(translator, tupleWords, modelVectors, size)
        
      #vectorDefinition=np.divide(vectorDefinition,lenListWordsDefinition)
      try:
        simpleVectorDefinition=np.exp(vectorDefinition) / float(sum(np.exp(vectorDefinition)))
        localVectorsDefinitions[definition]=[simpleVectorDefinition,vectorDefinition1,vectorDefinition2]
        #import pdb; pdb.set_trace()
      except Warning as e:
        print(vectorDefinition)
        import pdb; pdb.set_trace()
        
    
    vectorsDefinitions[word]=localVectorsDefinitions
  ######################################################  
    
  sleep(0.01)
  #######################################################
  
  print("filled vectors def",datetime.datetime.now())
  return 100#at.shape[1]#len(pca)

    
if ('dictTrain' not in locals()):
  exec(open('parseTrainFile.py').read())
#import pdb; pdb.set_trace()


if ('senseDict' not in locals()):
  exec(open('parsDic.py').read())

  
print("parsed files!")

parser = ConfigParser()
parser.read('conf.ini')

size=int(parser.get('parameters','sizeVectors'))

lenSize=2

if ('modelVectors' not in locals()):
  
  if(100==size):
    modelVectors=gensim.models.KeyedVectors.load_word2vec_format("~/bnc-word2vec/vectors_nnet_100.txt",binary=False)
    
  elif(300==size):
    modelVectors=gensim.models.KeyedVectors.load_word2vec_format("~/GoogleNews-vectors-negative300.bin",binary=True)
    
  elif(500==size):
    modelVectors=gensim.models.KeyedVectors.load_word2vec_format("~/bnc-word2vec/vectors_nnet_500.txt",binary=False)
    
  #modelVectors=gensim.models.KeyedVectors.load_word2vec_format("/extra/arocha/representaciones vectoriales/word2vec/vectorsEn.bin",binary=False)


print(datetime.datetime.now())

filterWordsMainTest=False

listStopWords=stopwords.words('english')
tupleStopWords=tuple(listStopWords)

vectorsTrain={}
vectorsDefinitions={}

sizeVDefinition=fillVectorsDefinition(tupleStopWords, vectorsDefinitions, senseDict, modelVectors, filterWordsMainTest, size)
fillVectorsTrain(tupleStopWords, vectorsTrain, dictTrain, modelVectors, filterWordsMainTest, size)


print(datetime.datetime.now())

sleep(1)
arrayTest=[]
maxShape=0
ii=0
#import pdb; pdb.set_trace()
for key in sorted(vectorsTrain):
  listFields=key.split('.')
  idSenses=dictTrain[key]['answers']
  posibleSenses=senseDict[key.split('.')[0]].keys()
  
  for idSense in posibleSenses:
    #import pdb; pdb.set_trace()  
    if(idSense in idSenses):
      arrayTest.append((vectorsTrain[key],vectorsDefinitions[listFields[0]][idSense][1],vectorsDefinitions[listFields[0]][idSense][2],[1,0]))
    else:
      arrayTest.append((vectorsTrain[key],vectorsDefinitions[listFields[0]][idSense][1],vectorsDefinitions[listFields[0]][idSense][2],[0,1]))
  #import pdb; pdb.set_trace()  
      #arrayTest.append((vectorsTrain[key],vectorsDefinitions[listFields[0]][idSense]))
    

#import pdb; pdb.set_trace()
random.shuffle(arrayTest)
#import pdb; pdb.set_trace()
vectorsTrain=None

print ("after create arrayTest")
shapeInput=size
print ("shape")
ir1=0
lenD=10
zd=np.zeros((len(arrayTest),lenSize,size))
zd2=np.zeros((len(arrayTest),size))
zd3=np.zeros((len(arrayTest),lenSize,size))

zd4=np.zeros((len(arrayTest),lenD,size))
zd5=np.zeros((len(arrayTest),lenD,size))
zd6=np.zeros((len(arrayTest),2))

zdw=np.zeros((len(arrayTest),500))

sleep(1)
print("after put zd and zd2")

for trainCase in arrayTest:
  #import pdb; pdb.set_trace()
  try:    
    zd[ir1]=trainCase[0][0] #c.i.
    zd2[ir1]=trainCase[0][2]#palabra central
    zd3[ir1]=trainCase[0][1]#c.d.

    zdw[ir1,0:100]=trainCase[0][0][0]
    zdw[ir1,100:200]=trainCase[0][0][1]
    zdw[ir1,200:300]=trainCase[0][2]
    zdw[ir1,300:400]=trainCase[0][1][0]#.resize(shapeInput[0],maxShape)
    zdw[ir1,400:500]=trainCase[0][1][1]
        
    zd4[ir1]=trainCase[1]    
    zd5[ir1]=trainCase[2]

    
    zd6[ir1]=trainCase[3]
    #import pdb; pdb.set_trace()
    
  except:
    import pdb; pdb.set_trace()
    print("exception",trainCase[0][0].shape,trainCase[0][1].shape,trainCase[1].shape,ir1)
    #bre
  ir1+=1

epochs=int(parser.get('parameters','epochs'))
batchSize=int(parser.get('parameters','batchSize')) 
  
finalModel=buildModel(size,10)
fitModel([zdw,zd4,zd5,zd6], finalModel, batchSize, epochs)


