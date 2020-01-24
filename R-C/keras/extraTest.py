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
#from mainMakeArrayTrain import fillVectorsDefinition


def fillVectorsDefinition(vectorsDefinitions, senseDict,modelVectors,filterWordsMainTest,size):

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
        
      if(filterWordsMainTest==True):
        filteredWordsD=[word for word in listWords if word not in tupleStopWords]          
        tupleFilteredWordsD=tuple(filteredWordsD)
        vectorDefinition=mergeVectors(translator, tupleFilteredWordsD, modelVectors, size)

      else:
        tupleWords=tuple(listWords)
        vectorDefinition=mergeVectors(translator, tupleWords, modelVectors, size)
        vectorDefinition1=formList(instanceDefinition,translator,modelVectors,10)
        vectorDefinition2=formList(instanceDefinition,translator,modelVectors,10,True)
        #vectorDefinition=np.exp(vectorDefinition) / float(sum(np.exp(vectorDefinition)))#vectorDefinition.clip(0)
        differentZero=0
        averageVector=0

        #vectorDefinition=pca.fit_transform(vectorDefinition)
        bt=np.copy(vectorDefinition)
        vectorDefinitionL=np.zeros(size)
        index1=np.argmax(vectorDefinition)
        vectorDefinitionL[index1]=100
        #vectorDefinition=np.delete(vectorDefinition,index1)
        index2=np.argmax(vectorDefinition)
        if(index2>=index1):
           index2+=1
        #vectorDefinitionL[index2]=40

        #vectorDefinition=pca.fit_transform(vectorDefinition)
        #vectorDefinition=np.divide(vectorDefinition,lenListWordsDefinition)
        simpleVectorDefinition=np.exp(vectorDefinition) / float(sum(np.exp(vectorDefinition)))
        localVectorsDefinitions[definition]=[simpleVectorDefinition,vectorDefinition1,vectorDefinition2]
#        localVectorsDefinitions[definition]=np.exp(vectorDefinition) / float(sum(np.exp(vectorDefinition)))#vectorDefinition np.power((vectorsDefinitions*2),2)(vectorDefinition*25)
    
    vectorsDefinitions[word]=localVectorsDefinitions
  ######################################################
      
  ################################
    
  sleep(0.01)
  #######################################################
  
  print("filled vectors def",datetime.datetime.now())
  return 100#at.shape[1]#len(pca)



def writeAnswersFile(stringAnswer):
 with open('testExtra.text','a') as fileTest:
  fileTest.write(stringAnswer+'\n')

def writeAnswersFile2(stringAnswer):
 with open('test2Extra.text','a') as fileTest:
  fileTest.write(stringAnswer+'\n')
###################################


if(os.path.exists('testExtra.text')==True):
 os.remove('testExtra.text') 

if(os.path.exists('test2Extra.text')==True):
 os.remove('test2Extra.text') 
 
#if ('senseDict' not in locals()):
# exec(open('parsDic.py').read())

#exec(open('parsDicExtra.py').read())
exec(open('parsDicExtra.py').read())
nSenseDict=createExtraDict()

sizeVectors=100
nVectorsDefinitions={}
fillVectorsDefinition(nVectorsDefinitions, nSenseDict, modelVectors,False,sizeVectors)

exec(open('parseTest2.py').read())
newQuestions=openAndLoadTestFileExtra()

#if ('dictTest' not in locals()):
# dictQuestions=openAndLoadTestFile()
 #exec(open('parsTest.py').read())

listStopWords=stopwords.words('english')
tupleStopWords=tuple(listStopWords)
#fillVectorsDefinition(tupleStopWords, vectorsDefinitions, senseDict, modelVectors, filterWordsMainTest, size)

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

for testCase in sorted(newQuestions):

 if(testCase.split('.')[0] not in nSenseDict):
  continue
 
 wordsTesting=newQuestions[testCase] 
 #import pdb; pdb.set_trace()
 listOrations=splitContextHead(wordsTesting)#wordsTesting['context'].split(wordsTesting['target'])
 testC={}
 splitOrationsAndPadB(testCase, listOrations, testC, sizeLen, modelVectors, sizeVectors)
 #import pdb; pdb.set_trace()
 #print(testC,len(testC),len(testC['activate.v.bnc.00008457']),type(testC['activate.v.bnc.00008457'][0]),testC['activate.v.bnc.00008457'][1].shape)4
 keysV=tuple(testC.keys())

 #break
 ##########################
 #print(keysV,testCase)
 
 if(wordTest!=testCase.split('.')[0]):
  
  wordTest=testCase.split('.')[0]
  print("palabra:"+wordTest, )
  #import pdb; pdb.set_trace()
  definitions=nSenseDict[wordTest]
  testVectorsDefinitions={}
  testVectorsDefinitions=nVectorsDefinitions[wordTest]
  #print(definitions)

 ##########################
 
 #vectorCase=np.empty([100,1],dtype=np.float32)


 #if(wordTest=='activate'):
 try:
  xtestB[0,:,:]=testC[keysV[0]][0]
 except:
  print(listOrations[0])
  xtestB[0,:,:]=np.zeros((sizeLen,sizeVectors))
  #import pdb; pdb.set_trace()
  
 try:
  xtestF[0,:,:]=testC[keysV[0]][1]
 except:
  print(listOrations[1])
  xtestF[0,:,:]=np.zeros((sizeLen,sizeVectors))
  #import pdb; pdb.set_trace()
 #  sumv=np.divide(sumv,index)
 #xtest[:,0]=sumv#vectorCase
 #
 #############################
 
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
 if(len(posibleValues)>0):
  complementAnswer=''
  for i in posibleValues:
   complementAnswer=complementAnswer+i+' '
  stringAnswer=""
  stringAnswer=testCase.split('.')[0]+' '+testCase+' '+complementAnswer
  writeAnswersFile(stringAnswer)
 #################################################
 counter+=1 
 ##############################

  
