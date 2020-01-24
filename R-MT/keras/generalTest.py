import gensim
import numpy as np
import nltk
import string
from numpy import linalg as LA
import os
import os.path
from nltk.corpus import stopwords
from vectorOperations import sumVectors, getVectorModel, distanciaE, similitud, convertSimiliratyEuclidian, getListVectors
from parsTest import *
from parserTools import *

def writeAnswersFile(stringAnswer):
 with open('test.text','a') as fileTest:
  fileTest.write(stringAnswer+'\n')

###################################


if(os.path.exists('test.text')==True):
 os.remove('test.text') 

if ('senseDict' not in locals()):
 exec(open('parsDic.py').read())
 
if ('dictTest' not in locals()):
 dictQuestions=openAndLoadTestFile()

parser = ConfigParser()
parser.read('conf.ini')

sizeVectors=int(parser.get('parameters','sizeVectors'))

if ('modelVectors' not in locals()):
 if(100==sizeVectors):
  modelVectors=gensim.models.KeyedVectors.load_word2vec_format("~/bnc-word2vec/vectors_nnet_100.txt",binary=False)
 elif(300==sizeVectors):
  modelVectors=gensim.models.KeyedVectors.load_word2vec_format("~/GoogleNews-vectors-negative300.bin",binary=True)
 elif(500==sizeVectors):
  modelVectors=gensim.models.KeyedVectors.load_word2vec_format("~/bnc-word2vec/vectors_nnet_500.txt",binary=False)
 

print ("begins feed model NN")

batchSize=1

sizeLen=int(parser.get('parameters','sizeLen'))

xtestF = np.zeros([batchSize,sizeLen,sizeVectors], dtype=np.float32)
xtestB = np.zeros([batchSize,sizeLen,sizeVectors], dtype=np.float32)

listStopWords=stopwords.words('english')
tupleStopWords=tuple(listStopWords)

filterWords=False
stopTest=0

sleep(0.1)
translator=str.maketrans('','',string.punctuation)
#print("after translator")
wordTesting=""
wordTest=""
counter=0
definitions=[]
testVectorsDefinitions={}

for testCase in sorted(dictQuestions):

 wordsTesting=dictQuestions[testCase] 
 
 listOrations=splitContextTarget(wordsTesting)#wordsTesting['context'].split(wordsTesting['target'])
 testC={}
 splitOrationsAndPadB(testCase, listOrations, testC, sizeLen, modelVectors, sizeVectors)

 keysV=tuple(testC.keys())


 ##########################

 if(wordTest!=testCase.split('.')[0]):
  
  wordTest=testCase.split('.')[0]
  print("palabra:"+wordTest, )
  definitions=senseDict[wordTest]
  testVectorsDefinitions={}
  testVectorsDefinitions=vectorsDefinitions[wordTest]

 ##########################


 try:
  xtestB[0,:,:]=testC[keysV[0]][0]
 except:
  print(listOrations[0])
  xtestB[0,:,:]=np.zeros((sizeLen,sizeVectors))

  
 try:
  xtestF[0,:,:]=testC[keysV[0]][1]
 except:
  print(listOrations[1])
  xtestF[0,:,:]=np.zeros((sizeLen,sizeVectors))

 #
 #############################
 
 
 

 extraInfo=np.zeros((sizeVectors,sizeVectors))
 extraInfo[0,:]=getVectorModel(wordTest,modelVectors)
 newVectorCase= finalModel.predict([xtestB,xtestF,extraInfo])
 arrayOutput=np.copy(newVectorCase[0])#.get()
 

 localSimilitudes={} 

 maxId='U'
 
 for key,value in testVectorsDefinitions.items():
  bufferValue=np.copy(value)  
  indexMax=np.argmax(bufferValue)#(LA.norm(arrayOutput-value))
  indexMaxO=np.argmax(arrayOutput)

  if(indexMax==indexMaxO):#localSimilitudes[key]):   
   maxId=' '+key
   break

 stringAnswer=""
 stringAnswer=testCase.split('.')[0]+"."+testCase.split('.')[1]+' '+testCase+' '+maxId
 
 writeAnswersFile(stringAnswer)
 #################################################
 counter+=1 
 ##############################

  
