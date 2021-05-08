# create metadata
import os
from tqdm import tqdm

import h5py

CURR_DIR = os.getcwd()
LABEL = "CAR"

def main() :

	path = os.path.join(CURR_DIR , "02958343")

	with open( 'model_names_{}.txt'.format(LABEL) , 'w') as file :
		loop = os.listdir(path)

		for _ in tqdm(loop) :

			file.write("{}\n".format(_))

if __name__ == '__main__':
	main()
