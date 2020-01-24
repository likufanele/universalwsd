import string
import numpy as np
from numpy import linalg as LA
from scipy.spatial import distance
import math
from glove import *
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer

def getVectorModel(i,modelV,size=100,translator=None):
 #import pdb; pdb.set_trace()
 if(translator):
  newWord=i.translate(translator)
 else:  
  newWord=i
  
 #print(newWord,type(counter))
 #print(newWord not in string.punctuation)
 
 if (newWord not in string.punctuation)and(newWord):
  normalizedString=newWord.lower()
  
  if((newWord in modelV.vocab)==True):
   #print('in',newWord)
   #normVector=LA.norm(modelV[newWord])
   return modelV[newWord]
      
  elif ((normalizedString in modelV.vocab)==True):
   #normVector=LA.norm(modelV[normalizedString])
   #print('in',normalizedString)
   return modelV[normalizedString]
   

  else:
   
   if((normalizedString[-1]=='s')or(normalizedString[-1]=='d')):
    cuttedWord=normalizedString[0:-1]
    if(cuttedWord in modelV.vocab):
     #print(cuttedWord)
     return modelV[cuttedWord]
   
   
   porter_stemmer = PorterStemmer()
   steamedWordPorter = porter_stemmer.stem(newWord)

   
   if(steamedWordPorter in modelV.vocab):
    #print(steamedWordPorter)
    return modelV[steamedWordPorter]

   
   snowball_stemmer = SnowballStemmer('english')
   steamedWordSnowball=snowball_stemmer.stem(newWord)
   
   if(steamedWordSnowball in modelV.vocab):
    #print(steamedWordSnowball)
    return modelV[steamedWordSnowball]

   wordnet_lemmatizer = WordNetLemmatizer()
   lemmatizedWordNet=wordnet_lemmatizer.lemmatize(newWord)
   
   if(lemmatizedWordNet in modelV.vocab):
    #print(lemmatizedWordNet)
    return modelV[lemmatizedWordNet]
    
   ###########################
     
   
 return np.zeros(size).astype(np.float32) #np.zeros((300)),0

#########################################################

def mergeMaxItems(npArray1,npArray2):
    return np.maximum(npArray1,npArray2)

#########################################################

def mergeVectors(translator,listS,modelV,size): 
  modelS=np.zeros((size)).astype(np.float32)
  counterWord=0
  #type(counterWord)
  #print ("mV")
  #type(listS)
  #import pdb; pdb.set_trace()
  for i in listS:
    #print(i)
    if(i=="'s"):
        i="is"
    elif(i=="'re"):
        i="are"
            
    newWord=i.translate(translator)
       
    tupleReturn=getVectorModel(newWord,modelV,size,translator)
    modelS=sumVector(modelS,tupleReturn)
    if(LA.norm(tupleReturn)>0):
     counterWord+=1
    #print(i,tupleReturn)
    ############################

  #print(counterWord)
  
  #if(average):
   #print (average)
  # modelS=np.divide(modelS,counterWord)
  return modelS
###################################


def sumVector(npArray1,npArray2):
 #try:
 # (npArray1+npArray2)
 #except:
 # import pdb; pdb.set_trace()
 return (npArray1+npArray2)

def sumVectors(translator,listS,modelV,size=300):
 modelS=np.zeros((size))
 #print("sumVectors",listS,len(listS))
 counterWords=0
 for i in listS: 
  if(i=="'s"):
   i="is"
  elif(i=="'re"):
   i="are"
   
  newWord=i.translate(translator)    
  if(modelS.size!=0):
    modelS=modelS+getVectorModel(newWord,modelV,size)#model[newWord]
  else:
   modelS=getVectorModel(newWord,modelV)#model[newWord]
  #print(newWord,model[newWord][0],modelS[0])  
  #input("Press Enter to continue...")
  return modelS

#########################################################

def getListVectors(tupleStrings,translator,modelV,size):
 newListVectors=np.array([]).astype(np.float32)#np.zeros(()).astype(np.float32)
 
 for word in tupleStrings:
  if(word=="'s"):
   word="is"
  elif(word=="'re"):
   word="are"

  word=word.lower()
  newWord=word.translate(translator)
  
#  if((not newWord)==False):
  nVector=getVectorModel(newWord,modelV,size)

  
  if(LA.norm(nVector)>0):
   #print(nVector,newListVectors)
   
   if(newListVectors.size!=0):    
    #print(newListVectors.size,nVector.size)
    newListVectors=np.row_stack([newListVectors,nVector])
    #print("added")
   else:
    newListVectors=nVector
    #print("seted",newListVectors)
    
 #print(newListVectors,newListVectors.shape) 
 return newListVectors


def similitud(modelS1,modelS2):
 arriba=np.dot(modelS1,modelS2)
 #print(arriba)
 abajo=LA.norm(modelS1)*LA.norm(modelS2)
 #print(abajo)
 return (arriba/abajo)


def distanciaE(modelS1,modelS2):
 return distance.euclidean(modelS1,modelS2)


   
def convertSimiliratyEuclidian(correlation):
 return math.sqrt(2*(1-correlation))
 
