import numpy as np
import cv2
import os

path='Data/Petimages'
subFolders=['Cat','Dog']

print('Total no of files before removing corrupted files')
print(len(os.listdir('Data/Petimages/Cat'))+len(os.listdir('Data/Petimages/Dog')))

corrupted=0
for currentFolder in subFolders:
    currentPath=os.path.join(path,currentFolder)
    for img in os.listdir(currentPath):
        try:
            img_array = cv2.imread(os.path.join(currentPath,img)  ,cv2.IMREAD_GRAYSCALE)  
            new_array = cv2.resize(img_array, (100, 100))  
        except Exception as e:  
                print(os.path.join(currentPath,img))
                corrupted=corrupted+1
                os.remove(os.path.join(currentPath,img))

print("Total no of corrupted files {}".format(corrupted))  
print('Total no of files After removing corrupted files')
print(len(os.listdir('Data/Petimages/Cat'))+len(os.listdir('Data/Petimages/Dog')))
        