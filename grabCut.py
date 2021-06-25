import numpy as np
import cv2
from matplotlib import pyplot as plt

def grabCut(img = "plane3.jpg"):
  img = cv2.imread(img)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  plt.imshow(img),plt.show()

  mask = np.zeros(img.shape[:2],np.uint8)
  bgdModel = np.zeros((1,65),np.float64)
  fgdModel = np.zeros((1,65),np.float64)
  rect = (30,50,350,180)
  
  cv2.grabCut(img,mask,rect,bgdModel,fgdModel,100,cv2.GC_INIT_WITH_RECT)

  mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
  img = img*mask2[:,:,np.newaxis]
  img[img==0] = 255

  plt.imshow(img),plt.show()

grabCut()
