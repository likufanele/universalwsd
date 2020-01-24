entrieDict=z['activate']
type(entrieDict[1][0])
palabrasCadena=entrieDict[1][0].split()

for i in palabrasCadena: 
  if (i not in string.punctuation):  
    normalizedString=i.lower()      
    if(normalizedString in model):
      print (model[normalizedString])