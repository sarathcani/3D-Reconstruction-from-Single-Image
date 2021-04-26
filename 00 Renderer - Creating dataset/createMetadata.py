# create metadata
import os
from tqdm import tqdm

CURR_DIR = os.getcwd()
DATA_DIR = os.path.join(CURR_DIR , "data" , "car")
LABEL = "CAR"
FILENAME = "ShapenetRendering_{}.txt".format(LABEL)

def main() :

	counter = 0
	with open( FILENAME , "w") as file :

		for model in tqdm(os.listdir(DATA_DIR)) :
			for i in range(24) :
				file.write("{}_{}\n".format(model , i))
				counter += 1

	print("{} lines added to {}".format(counter , FILENAME) )
	
if __name__ == '__main__':
	main()
