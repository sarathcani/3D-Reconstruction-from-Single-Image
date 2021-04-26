### Visualize model
from tensorflow.keras.utils import plot_model
from resnet import Resnet50

def main() :
	resnet = Resnet50()

	model = resnet.get_model()

	plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

if __name__ == '__main__':
	main()