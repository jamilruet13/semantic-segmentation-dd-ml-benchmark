#importing dependencies
import cv2
import numpy as np
from sklearn import mixture
from scipy import linalg
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
from time import time
from sklearn import preprocessing
from sklearn.utils import shuffle
from itertools import chain
from skimage import feature
import os


color_iter = itertools.cycle(['purple', 'Orange', 'pink', 'red', 'white','Green','Blue'])

# dictionary of color codes for creating segmentation masks
_color_codes = {
  1 : (180, 30, 145), #  -->purple
  2 : (48, 130, 245), #  --> Orange
  3 : (255, 0, 255), #  --> pink
  4 : (75, 25, 230), #  -->red
  5 : (255, 255, 255), #  --> white
  6 : (75, 180, 60), #  -->Green
  7 : (200, 130, 0) #  --> Blue

}

def test(imtest, gmm):

			lab1=gmm.predict(imtest)
			return lab1

#---------Runs GMM Fit on Each Random Combination of 1000 Points, 'num_patches' number of times---------#

def train(num_patches, image,n_samples,w,h):
  np.random.seed(1)
  for i in range(1, num_patches): #Fit a Gaussian mixture with EM using five components repeatedly with small  random samples from the data

	  imtrain = shuffle(image,random_state=None)
	  imtrain=imtrain[:1000]

	  t=time()
	  gmm = mixture.GaussianMixture(n_components=7, covariance_type='full', random_state=None, 
				tol=0.001, reg_covar=1e-06, max_iter=1200, n_init=1, init_params='kmeans', 
				warm_start=True).fit(imtrain)
  
  print ("Gaussian Mixture Done in %0.3fs." % (time() - t))
  return gmm


def segmented(image,samples,label, num_comp,w,h):
  #Add dimension to [n,] array
	labels=np.expand_dims(label, axis=0)
	labels=np.transpose(labels)

	for i in range(1,num_comp):

		indices=np.where(np.all(labels==i, axis=-1))
		indices = np.unravel_index(indices, (w,h), order='C')
		type(indices)
		indices=np.transpose(indices)

		l = chain.from_iterable(zip(*indices))

		for j, (lowercase, uppercase) in enumerate(l):
				# set the colour accordingly

				image[lowercase,uppercase] = _color_codes[(i)]

	return image

def createData(image, n_samples,d):
	#Intialisation for Local Binary Patterns Descriptor
	numPoints = 24
	#Number of samples per component
	radius = 8
	img_src = cv2.GaussianBlur(image,(5,5),0) #

	imtest=cv2.cvtColor(img_src, cv2.COLOR_BGR2LAB)
	img_gray= cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

	lbp = feature.local_binary_pattern(img_gray, numPoints,
		radius, method="uniform")

	lbp=np.reshape(lbp,(n_samples,1))

	imtest= np.reshape(imtest, (n_samples, d))
	data=np.column_stack((imtest, lbp))

	data= preprocessing.normalize(imtest, norm= 'l2')
	return data, imtest

def runcluster(dataset,img,basedir='./predictions_cluster'):
		img_src = f'{dataset}/images/{img}-ortho.tif'
		print(img)
		img_src = cv2.imread(img_src)
		cv2.waitKey(0)
		predsfile = os.path.join(basedir, f'{img}-prediction.png')

		print(predsfile)
		w, h, d = img_src.shape
		assert d == 3	
		# Number of samples per component
		n_samples = w*h
		#Number of sets of training samples
		num_patches=100;	
		print(w,h)
		
		samples, imtest=createData(img_src, n_samples,d)
		#CallTrainStep
		gmm =train(num_patches, samples ,n_samples,w,h)
		#prepimage(imtest, num_patches)
		
		#Calculate Labels by Testing
		lab1 =test(samples, gmm)
			
		#CallSegmentation
		seg1=segmented(img_src,samples,lab1, 7,w,h)
		
		#Save Image	
		cv2.imwrite(predsfile,seg1)
		return seg1
