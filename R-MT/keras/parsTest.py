import nltk
import nltk.corpus.reader.senseval as sensP
#import xml.etree.ElementTree as ET
import re
from time import sleep
from lxml import html
from bs4 import Tag
from bs4 import BeautifulSoup

def openAndLoadTestFile():
 with open('./../EnglishLS.test/EnglishLS.test','rt') as fd:
  dicXml=fd.read()

 # doc=html.fromstring(dicXml)
 # listContexts=[]

 dictQuestions={}
  
 #doc = html.fromstring(trainXml)
 soup = BeautifulSoup(dicXml, 'html.parser')  
 
 lexelts=soup.find_all('lexelt')
 for lexelt in lexelts:
   #import pdb; pdb.set_trace()
  instancesLocal=lexelt.find_all('instance')
  for instance in instancesLocal:
   #import pdb; pdb.set_trace()
   if(type(instance)==Tag):
    #print(type()
    dictItem=instance.attrs
    #print(dictItem['id'])
    exampleDict={}   
    exampleDict['text']=instance.text.strip()
    exampleDict['target']=instance.head.text
    exampleDict['context']=str(instance.context)
    exampleDict['head']=str(instance.head)
    dictQuestions[dictItem['id']]=exampleDict
    #    dictAnswers[answerAttrs['instance']]
  
 print ("end script parsTest.py(openAndLoadTestFile)")
 return dictQuestions

def splitContextTarget(itemDict):
  listOrations=itemDict['context'].split(itemDict['head'])
  listOrations[0]= re.sub('<[^>]*>', '',listOrations[0]).rstrip()

  listOrations[1]= re.sub('<[^>]*>', '',listOrations[1]).rstrip()

  if(len(listOrations)>2):
   import pdb; pdb.set_trace()
   listOrations[2]= re.sub('<[^>]*>', '',listsWords[2]).rstrip()
   complementTarget = re.sub('<[^>]*>', '',itemDict['head']).rstrip()
   listsWords[1]=(listsWords[1]+' '+complementTarget+' '+listsWords[2])
  
  return listOrations
  #str(instance.context).split(str(instance.head)))
  
