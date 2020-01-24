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
from parserTools import *
import re
from sequence import *

from configparser import ConfigParser

def fillVectorsTrain(tupleStopWords, vectorsTrain, dictTrain, modelVectors, filterWordsMainTest,size):
  print("filled vectors train",datetime.datetime.now())  
  translator=str.maketrans('','',string.punctuation)
  counter=0
  parser = ConfigParser()
  parser.read('conf.ini')
  
  lenSize=int(parser.get('parameters','sizeLen'))
  
  for key in sorted(dictTrain):
    idTestStringL=key.split('.')
    
    dictExercise=dictTrain[key]
    listsWords=dictExercise['contextTag'].split(dictExercise['head']) #sentence.replace(" n't","n't")
    
    listsWords[0]= re.sub('<[^>]*>', '',listsWords[0]).rstrip()
    listsWords[1]= re.sub('<[^>]*>', '',listsWords[1]).rstrip()
    
    if(len(listsWords)>2):

      listsWords[2]= re.sub('<[^>]*>', '',listsWords[2]).rstrip()
      complementTarget = re.sub('<[^>]*>', '',dictExercise['head']).rstrip()
      listsWords[1]=(listsWords[1]+' '+complementTarget+' '+listsWords[2])
      if(len(listsWords)>3):
        listsWords[3]= re.sub('<[^>]*>', '',listsWords[3]).rstrip()
        listsWords[1]=(listsWords[1]+' '+complementTarget+' '+listsWords[3])

        
    if(filterWordsMainTest==True):
      filteredWords=[word for word in listWords if word not in tupleStopWords]      
      tupleWords=tuple(filteredWords)
      newVectorDef=mergeVectors(translator,tupleWords,modelVectors,size)#,idTestStringL[0])
      vectorsTrain[key]=newVectorDef
      
    else:
      
      splitOrationsAndPadB(key,listsWords,vectorsTrain, lenSize, modelVectors,size)

      
    counter+=1
  
  print("filled vectors train",datetime.datetime.now())
  

  
def fillVectorsDefinition(tupleStopWords,vectorsDefinitions, senseDict,modelVectors,filterWordsMainTest,size):

  print("filled vectors def",datetime.datetime.now())
  translator=str.maketrans('','',string.punctuation)
  counter=0
  parser = ConfigParser()
  parser.read('conf.ini')
  
  for word in senseDict:
    definitions=senseDict[word]
    localVectorsDefinitions={}
    for definition in definitions:
      
      instanceDefinition=definitions[definition]
      instanceDefinition=instanceDefinition.replace(" n't","n't")
      listWords=instanceDefinition.split()

        
      if(filterWordsMainTest==True):
        filteredWordsD=[word for word in listWords if word not in tupleStopWords]          
        tupleFilteredWordsD=tuple(filteredWordsD)

        #print(filteredWordsD)
        vectorDefinition=mergeVectors(translator, tupleFilteredWordsD, modelVectors, size)
          

      else:
        tupleWords=tuple(listWords)

        vectorDefinition=mergeVectors(translator, tupleWords, modelVectors, size)
        
        differentZero=0
        averageVector=0
        
        
        bt=np.copy(vectorDefinition)
        
        vectorDefinitionL=np.zeros(size)
        index1=np.argmax(vectorDefinition)
        vectorDefinitionL[index1]=100
        vectorDefinition=np.delete(vectorDefinition,index1)
        index2=np.argmax(vectorDefinition)
        if(index2>=index1):
           index2+=1
        vectorDefinitionL[index2]=75
        
      
      localVectorsDefinitions[definition]=vectorDefinitionL#np.exp(vectorDefinition) / float(sum(np.exp(vectorDefinition)))#vectorDefinition np.power((vectorsDefinitions*2),2)(vectorDefinition*25)
    
    vectorsDefinitions[word]=localVectorsDefinitions
  ######################################################
    
  sleep(0.01)
  #######################################################
  
  print("filled vectors def",datetime.datetime.now())
  return int(parser.get('parameters','sizeVectors'))

    
if ('dictTrain' not in locals()):
  exec(open('parseTrainFile.py').read())


if ('senseDict' not in locals()):
  exec(open('parsDic.py').read())

print("parsed files!")

parser = ConfigParser()
parser.read('conf.ini')

size=int(parser.get('parameters','sizeVectors'))

lenSize=int(parser.get('parameters','sizeLen'))

if ('modelVectors' not in locals()):
  if(100==size):
    modelVectors=gensim.models.KeyedVectors.load_word2vec_format("~/bnc-word2vec/vectors_nnet_100.txt",binary=False)
  elif(300==size):
    modelVectors=gensim.models.KeyedVectors.load_word2vec_format("~/GoogleNews-vectors-negative300.bin",binary=True)
  elif(500==size):
    modelVectors=gensim.models.KeyedVectors.load_word2vec_format("~/bnc-word2vec/vectors_nnet_500.txt",binary=False)
    



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
  idSense=dictTrain[key]['answer']
  
  if(idSense!='U'):
    arrayTest.append((vectorsTrain[key],vectorsDefinitions[listFields[0]][idSense]))
    

random.shuffle(arrayTest)
vectorsTrain=None


print ("after create arrayTest")
shapeInput=size
print ("shape")
ir1=0

zd=np.zeros((len(arrayTest),lenSize,size))
zd2=np.zeros((len(arrayTest),lenSize,size))
zd3=np.zeros((len(arrayTest),size))
zd4=np.zeros((len(arrayTest),sizeVDefinition))
sleep(1)
print("after put zd and zd2")

for trainCase in arrayTest:
  
  try:    
    zd[ir1]=trainCase[0][0]#.resize(shapeInput[0],maxShape)
    zd2[ir1]=trainCase[0][1]
    zd3[ir1]=trainCase[0][2]
    zd4[ir1]=trainCase[1]
  except:
    print("exception",trainCase[0][0].shape,trainCase[0][1].shape,trainCase[1].shape,ir1)
    #break
  ir1+=1



epochs=int(parser.get('parameters','epochs'))
batchSize=int(parser.get('parameters','batchSize'))
finalModel=buildModel(lenSize,size)
fitModel([zd,zd2,zd3,zd4], finalModel, batchSize, epochs)
