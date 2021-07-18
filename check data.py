import numpy as np
import cv2
import os
from tqdm import tqdm


path='Data/Petimages'
subFolders=['Cat','Dog']
i=0
for currentFolder in subFolders:
    currentPath=os.path.join(path,currentFolder)
    for img in os.listdir(currentPath):
        try:
            img_array = cv2.imread(os.path.join(currentPath,img)  ,cv2.IMREAD_GRAYSCALE)  # convert to array
            new_array = cv2.resize(img_array, (100, 100))  # resize to normalize data size
        except Exception as e:  # in the interest in keeping the output clean...
                print(os.path.join(currentPath,img))
                i=i+1
                os.remove(os.path.join(currentPath,img))

print(i)  
        

    