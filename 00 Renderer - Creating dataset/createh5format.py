### Script for creating the rendered data into hdf5 format
#
#
# Install 
# 	- h5py
# 	- numpy
# 	- imageio
# 	- tqdm (not necessary)
#
# Define
# 	- CURR_DIR
# 	- DATA_DIR (recommended to keep the data in './data')
# 	- LABEL
# 	- FILENAME


# Importing Libraries
import h5py
import os
import numpy as np 
import imageio
from tqdm import tqdm

### Define these
CURR_DIR = os.getcwd()
DATA_DIR = os.path.join(CURR_DIR , "data" , "car")
LABEL = "CAR"
FILENAME = "ShapenetRendering_{}.h5".format(LABEL)

IMG_NAMES = ['00.png', '01.png', '02.png', '03.png', '04.png', '05.png', '06.png', '07.png', '08.png', '09.png', '10.png', '11.png', '12.png', '13.png', '14.png', '15.png', '16.png', '17.png', '18.png', '19.png', '20.png', '21.png', '22.png', '23.png']
## -------------------------------------------------------

def main() :

	### This counter used for keeping track of the number of images/groups present in the h5 file
	counter = 0

	### Open the file with FILENAME
	with h5py.File(FILENAME , 'w') as hdf :

		### Loop through the models
		### For each image in the model create a group
		### with img and pose as dataset
		for model_name in tqdm(os.listdir(DATA_DIR)) :
			model_path = os.path.join(DATA_DIR , model_name , "rendering" )
			r_metadata_path = os.path.join(model_path , "rendering_metadata.txt")
			with open(r_metadata_path , 'r') as r_metadata  :
				for i in range(24) :

					group_name = "{}_{}".format(model_name , i)

					## Create a group in the h5 file with group_name
					group = hdf.create_group(group_name)

					line = r_metadata.readline()
					azim = float(line.split()[0])
					elev = float(line.split()[1])
					pose = np.array([azim , elev])
					img_path = os.path.join(model_path , IMG_NAMES[i])

					## import image
					## create it into a np array
					img = imageio.imread(img_path)

					group.create_dataset("image" , data = img)
					group.create_dataset("pose" , data = pose)

					counter += 1

	print("Successfully created h5 format with {} images".format(counter))


if __name__ == '__main__':
	main()