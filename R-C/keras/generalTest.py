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

def writeAnswersFile2(stringAnswer):
 with open('test2.text','a') as fileTest:
  fileTest.write(stringAnswer+'\n')
###################################


if(os.path.exists('test.text')==True):
 os.remove('test.text') 

if(os.path.exists('test2.text')==True):
 os.remove('test2.text') 
 
if ('senseDict' not in locals()):
 exec(open('parsDic.py').read())
 
if ('dictTest' not in locals()):
 dictQuestions=openAndLoadTestFile()
 #exec(open('parsTest.py').read())

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

sizeLen=2

xtestF = np.zeros([batchSize,sizeLen,sizeVectors], dtype=np.float32)
xtestB = np.zeros([batchSize,sizeLen,sizeVectors], dtype=np.float32)
defBuffer1= np.zeros([batchSize,10,sizeVectors], dtype=np.float32)
defBuffer2= np.zeros([batchSize,10,sizeVectors], dtype=np.float32)
perceptronArray=np.zeros((1,5*sizeVectors))

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
 #import pdb; pdb.set_trace()
 listOrations=splitContextTarget(wordsTesting)#wordsTesting['context'].split(wordsTesting['target'])
 testC={}
 splitOrationsAndPadB(testCase, listOrations, testC, sizeLen, modelVectors, sizeVectors)
 #print(testC)
 #print(testC,len(testC),len(testC['activate.v.bnc.00008457']),type(testC['activate.v.bnc.00008457'][0]),testC['activate.v.bnc.00008457'][1].shape)4
 keysV=tuple(testC.keys())

 #break
 ##########################
 #print(keysV,testCase)
 if(wordTest!=testCase.split('.')[0]):
  
  wordTest=testCase.split('.')[0]
  print("palabra:"+wordTest, )
  #import pdb; pdb.set_trace()
  definitions=senseDict[wordTest]
  testVectorsDefinitions={}
  testVectorsDefinitions=vectorsDefinitions[wordTest]
  #print(definitions)
 

 ##########################


 #if(wordTest=='activate'):
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
 #  sumv=np.divide(sumv,index)

 #############################  
 
 #import pdb; pdb.set_trace()
 
 perceptronArray[0,0:100]=xtestB[0,0,:]
 perceptronArray[0,100:200]=xtestB[0,1,:]

 perceptronArray[0,300:400]=xtestF[0,0,:]
 perceptronArray[0,400:500]=xtestF[0,1,:]

 extraInfo=np.zeros((sizeVectors,sizeVectors))
 extraInfo[0,:]=getVectorModel(wordTest,modelVectors)
 perceptronArray[0,200:300]=extraInfo[0,:]
 
 
 posibleValues=[]
 
 for key,value in testVectorsDefinitions.items():
  #import pdb; pdb.set_trace()  
  defBuffer1[0,:,:]=value[1]
  defBuffer2[0,:,:]=value[2]
  
  newVectorCase= finalModel.predict([perceptronArray,defBuffer1,defBuffer2])  
  arrayOutput=np.copy(newVectorCase[0])#.get()
  if(np.argmax(arrayOutput)==0):
   posibleValues.append(key)
  #import pdb; pdb.set_trace()  
 
 #arrayOutput[:,0]
 localSimilitudes={} 
 #for key,value in testVectorsDefinitions.items():
 # localSimilitudes[key]=(LA.norm(arrayOutput-value)) #localSimilitudes[key])#similitud(arrayOutput[:,0],value)

 #minimumId = min(localSimilitudes, key=localSimilitudes.get)
 #maxId='U' 

 if(len(posibleValues)>0):
  complementAnswer=''
  for i in posibleValues:
   complementAnswer=complementAnswer+i+' '

  stringAnswer=""
  stringAnswer=testCase.split('.')[0]+"."+testCase.split('.')[1]+' '+testCase+' '+complementAnswer
 
  writeAnswersFile(stringAnswer)
  #################################################
 counter+=1 
 ##############################

  
