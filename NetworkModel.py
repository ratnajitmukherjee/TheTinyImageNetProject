from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Flatten, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.regularizers import l2

class BuildNetworkModel:
    
    def __init__(self):
        print("\n Loading Network Model...")

    # Define the convolution layer
    def conv2d_bn(self, input, filter_size, kernel_size):
        weight_decay = 1e-4
        x = Conv2D(filter_size, (kernel_size, kernel_size), activation = 'linear', kernel_regularizer=l2(weight_decay), padding='same')(input)
        x = BatchNormalization(axis=-1)(x)
        x = LeakyReLU(alpha=0.3)(x)
        return x
    
    # Define the Maxpool Layers
    def maxpool_2d(self, input, pool_size, stride_size):
        if stride_size is None:
            stride_size = pool_size
        x = MaxPooling2D(pool_size=(pool_size, pool_size), strides=stride_size, padding='same')(input)
        return x

    # Build the Sequential model
    def buildSequentialModel(self, input_size, num_classes):
        # get input size externally
        input_layer = Input(input_size)
        x = self.conv2d_bn(input_layer, filter_size=64, kernel_size=3)
        x = self.conv2d_bn(x, filter_size=64, kernel_size=3)
        x = self.maxpool_2d(x, pool_size=2, stride_size=None)
        return x

if __name__ == '__main__':
    input_size = (64, 64, 3)
    buildSeqNetwork = BuildNetworkModel()
    model = buildSeqNetwork.buildSequentialModel(input_size, 200)
    model.summary()


