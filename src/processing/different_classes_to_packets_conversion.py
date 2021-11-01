#import os
#import xml.etree.ElementTree as ET
# assign directory
#directory = r'C:\Users\DELL\OneDrive\Desktop\100 image\100 image'
 
# iterate over files in
# that directory
#for filename in os.listdir(directory):
#   ff = os.path.join(directory, filename)
#    tree = ET.parse(ff)
#    root = tree.getroot()
#    # checking if it is a file
#    while(i<10):
#        if(root[i][0].text!='rack row' or 'complete rack'):
#            root[i][0].text='packets'
#            i=i+1
#    tree.write(ff,'w')





#import xml.etree.ElementTree as ET
#tree = ET.parse(r'C:\Users\DELL\OneDrive\Desktop\trial.xml')
#root = tree.getroot()
#i=6
#while(i<10):
#    if(root[i][0].text!='rack row' or 'complete rack'):
#        root[i][0].text='packets'
#        i=i+1
    

#tree.write('trial.xml')


import xml.etree.ElementTree as ET
import os

path = r'/media/premium/common-biscuit/main/src/src/data_preprocessing/original_dataset'
#dest_path = os.path.join(path, 'converted')
#os.mkdir(dest_path)
my_set=set()
for filename in os.listdir(path):
    if not filename.endswith('.xml'): continue
    fullname = os.path.join(path, filename)
    tree = ET.parse(fullname)
    root = tree.getroot()
    i=6
    
    for x in tree.findall('object'):
        my_set.add(root[i][0].text)
        if(root[i][0].text=='rack row' or root[i][0].text=='complete rack'):
            i=i+1
            continue
        else:
            root[i][0].text='packets'
        i=i+1
#    tree.write(dest_path,filename)
    tree.write(fullname)