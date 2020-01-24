
import nltk
import nltk.corpus.reader.senseval as sensP
#import xml.etree.ElementTree as ET
import re
from time import sleep
from lxml import html, etree
from bs4 import Tag
from bs4 import BeautifulSoup

replace_target = re.compile("""<head.*?>.*</head>""")
replace_newline = re.compile("""\n""")
replace_dot = re.compile("\.")
replace_cite = re.compile("'")
replace_frac = re.compile("[\d]*frac[\d]+")
replace_num = re.compile("\s\d+\s")
rm_context_tag = re.compile('<.{0,1}context>')
rm_cit_tag = re.compile('\[[eb]quo\]')
rm_markup = re.compile('\[.+?\]')
rm_misc = re.compile("[\[\]\$`()%/,\.:;-]")



def openAndLoadTestFileExtra():
 with open('./../senseval2/corpora/english-lex-sample/test/eng-lex-samp.evaluation.xml',encoding="ISO-8859-1") as fd:
  dicXml=fd.read().encode('utf-8')
 parser = etree.XMLParser(dtd_validation=True)#ns_clean=True, recover=True, encoding='utf-8')
 #dtd = etree.DTD(open('./../../senseval2/corpora/english-lex-sample/test/lexical-sample.dtd'))
 doc=etree.parse('./../senseval2/corpora/english-lex-sample/test/eng-lex-samp.evaluation.xml',parser=parser)
 
 # listContexts=[]
 #root = html.parse('./../../senseval2/corpora/english-lex-sample/test/eng-lex-samp.evaluation.xml').getroot()
 #import pdb; pdb.set_trace()
 dictQuestions={}
 instances = doc.findall('.//instance')

 

 for instance in instances:
    text=''
    context=None
    #import pdb; pdb.set_trace()
    for child in instance:
        if (child.tag == 'context'):
            context = etree.tostring(child)
    if(context):
        newContext=clean_context(context.decode("utf-8"))
        exampleDict={}
        exampleDict['target']=instance.get('id').split('.')[0]
        exampleDict['context']=newContext
        #print("stop")
        dictQuestions[instance.get('id')]=exampleDict
          
 #doc = html.fromstring(trainXml)
 # soup = BeautifulSoup(dicXml, 'xml')#html.parser')  
 
 # lexelts=soup.find_all('lexelt')
 # for lexelt in lexelts:
 #   #import pdb; pdb.set_trace()
 #  instancesLocal=lexelt.find_all('instance')
 #  for instance in instancesLocal:
 #   import pdb; pdb.set_trace()
 #   if(type(instance)==Tag):
 #    #print(type()
 #    dictItem=instance.attrs
 #    #print(dictItem['id'])
 #    #if(instance['id'].split('.')[1]=='a'):
 #    exampleDict={}   
 #    exampleDict['text']=instance.text.strip()
 #    exampleDict['target']=instance.head.text
 #    exampleDict['context']=str(instance.context)
 #    exampleDict['head']=str(instance.head)
 #    dictQuestions[dictItem['id']]=exampleDict
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
  
def clean_context(ctx_in):
    ctx = replace_target.sub(' <target> ', str(ctx_in))
    ctx = replace_newline.sub(' ', str(ctx))  # (' <eop> ', ctx)
    ctx = replace_dot.sub(' ', str(ctx))     # .sub(' <eos> ', ctx)
    ctx = replace_cite.sub(' ', str(ctx))    # .sub(' <cite> ', ctx)
    ctx = replace_frac.sub(' <frac> ', str(ctx))
    ctx = replace_num.sub(' <number> ', str(ctx))
    ctx = rm_cit_tag.sub(' ', str(ctx))
    ctx = rm_context_tag.sub('', str(ctx))
    ctx = rm_markup.sub('', str(ctx))
    ctx = rm_misc.sub('', str(ctx))
    return ctx
