from lxml import html
from time import sleep

def removeSignsPunctuation(originalString):
    normalizedString=originalString.replace(',','')
    normalizedString=normalizedString.replace('.','')
    normalizedString=normalizedString.replace(')','')
    normalizedString=normalizedString.replace('(','')
    normalizedString=normalizedString.replace(';','')
    return normalizedString

with open('./../EnglishLS.train/EnglishLS.dictionary.xml','rt') as f:
    dicXml=f.read()
    
doc = html.fromstring(dicXml)

misc=""

senseDict={}

for lexelt in doc:

    listName=lexelt.items()

    stringName=listName[0][1]
    
    typeV='.v'in stringName
    typeN='.n' in stringName
    typeA='.a' in stringName
    
    stringName=stringName.replace('.v','')
    stringName=stringName.replace('.n','')
    stringName=stringName.replace('.a','')
    print("string:",stringName)
    
    sensesItem={}
    
    for senseLexelt in lexelt:
        misc=senseLexelt

        itemsSense=senseLexelt.items()
        senseElements=senseLexelt.values()
        
        listTuple=[]

        for itemList in (itemsSense):
            
            if(itemList[0]=='id'):
                listTuple.append(itemList[1])
                
            elif(itemList[0]=="gloss"):
                if (type(itemList[1]) is str)==True:
                    valueGloss=itemList[1]
                    valueGloss=removeSignsPunctuation(valueGloss)                        
                    listTuple.append(valueGloss)
                    
                elif (type(itemList[1]) is tuple)==True:
                    valueGlossTuple=itemList[1]
                    valueGloss=valueGlossTuple[1]
                    valueGloss=removeSignsPunctuation(valueGloss)                        
                    listTuple.append(valueGloss)
                    
            # if(value=="2555507"):
            #     print (valueGloss)
            #     break
        
        sensesItem[listTuple[0]]=listTuple[1]
    #--------------------------------------
    
    senseDict[stringName]=sensesItem
    #print (senseDict)
    #break
print ("end script parsDic.py")
