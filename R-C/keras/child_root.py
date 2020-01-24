for child_of_root in root:
  child_of_root.text
  child_of_root.attrib
  for tag in child_of_root:
    print (tag)
    sleep(1)
    print(tag.attrib)
    print(type(tag.attrib))
    print(tag.text)
    for tag in child_of_root:
      idTest=tag.attrib['id']
    print(idTest)
  sleep(5)