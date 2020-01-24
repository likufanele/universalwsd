#import nltk
#import nltk.corpus.reader.senseval as sensP
import xml.etree.ElementTree as ET
import re
from time import sleep


with open('./../EnglishLS.test/EnglishLS.test') as fd:
 doc=fd.read()

with open('./../EnglishLS.test/EnglishLS2.test') as fo:
 doc2=fo.read()

doP=sensP._fixXML(doc2)

tree = ET.ElementTree(ET.fromstring(doP))

lexeltPattern=r'<lexelt item=*(.*?)</lexelt>'
lexeltTuples=re.findall(lexeltPattern, doc, re.DOTALL)

contextPattern=r'<context>*(.*?)</context>'

listContexts=[]

for i in lexeltTuples:
 newContexts=[]
 newContexts=re.findall(contextPattern, i, re.DOTALL)
 listContexts.append(newContexts)

#listContexts=re.findall(contextPattern, doc, re.DOTALL)

orations=[]

for i in listContexts:
 localOrations=[]
 for j in i:
  buf=j.replace('<head>','')
  buf=buf.replace('</head>','')
  buf=buf.replace("\n",'')
  localOrations.append(buf)#=(localOrations+buf)
  #print (localOrations)
  #sleep(0.4)
 orations.append(localOrations)

root=tree.getroot()
  
listTargets=[]


for child_of_root in root:
  newDict=child_of_root.attrib
  originalString=newDict['item']
  string=originalString.replace('.v','')
  string=string.replace('.n','')
  string=string.replace('.a','')
  tupleTargets=(string,originalString)
  listTargets.append(tupleTargets)

  
setTest={}

for i,j in zip(listTargets,orations): 
 tupleTC=(i,j)
 setTest[i[0]]=tupleTC



#for target in z:
 #print (target)
 #print ("--------------------------------")
 #entrie=z[target]
 #for j in entrie[1]:
  #sleep(1.5)
  #print (j)
  #print ("********************************")
   

