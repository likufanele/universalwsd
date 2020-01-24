#from lxml import html, etree
from time import sleep
#from nltk.stem.lancaster import LancasterStemmer
from bs4 import Tag
from bs4 import BeautifulSoup


def removeSignsPunctuation(originalString):
    normalizedString=originalString.replace(',','')
    normalizedString=normalizedString.replace('.','')
    normalizedString=normalizedString.replace(')','')
    normalizedString=normalizedString.replace('(','')
    normalizedString=normalizedString.replace(';','')
    return normalizedString

def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])



with open('./../EnglishLS.train/EnglishLS.train','rt') as f:
 trainXml=f.read()
    
#doc = html.fromstring(trainXml)
soup = BeautifulSoup(trainXml, 'html.parser')

dictTrain={}

lexelts=soup.find_all('lexelt')

for lexelt in lexelts:   
    instancesLocal=lexelt.find_all('instance')
    for instance in instancesLocal:  
        if(type(instance)==Tag):
            answerAttrs=instance.answer.attrs
            answersKeys=[]
            for k in instance.find_all('answer'):
                if(k.attrs['senseid']!='U'):
                    answersKeys.append(k.attrs['senseid'])
            
            #import pdb; pdb.set_trace()
            if((answerAttrs['senseid']!='U')):#and(answerAttrs['instance'].split('.')[1]=='a')):
                exampleDict={}   
                #import pdb; pdb.set_trace()
                exampleDict['context']=instance.text.strip()
                exampleDict['target']=instance.head.text
                exampleDict['answers']=answersKeys#answerAttrs['senseid']
                exampleDict['head']=str(instance.head)
                exampleDict['contextTag']=str(instance.context)    
                dictTrain[answerAttrs['instance']]=exampleDict
    #import pdb; pdb.set_trace()
                #print(dictTrain)
                # break
    


            
print ("end script parsTestFile.py")
