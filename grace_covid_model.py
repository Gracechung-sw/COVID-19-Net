import tensorflow 
from tensorflow.keras.layers import Model
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.layers import Concatenate

def CovidNet(input_shape, num_classes):
    # article 'COVID-Net: A Tailored Deep Convolutional Neural Network Design for Detection of COVID-19 Cases from Chest Radiography Images'
    # page3 Network Architecture', They  makes network architecture heavy using lightweight residual 'projection-expresion-projection-extension(PEPX)' design pattern.
    def PEPX(input_tensor, filter_channels, name):
        pepx1 = Conv2D(filter_channels[0], (1, 1), activation='relu', name = name+'First-stage Projection')(input_tensor) #First-stage Projection: 1x1 convolutions for projecting input features to a lower dimension
        pepx2 = Conv2D(filter_channels[1], (1, 1), activation='relu', name = name+'Expansion')(pepx1) #Expansion: 1x1 convolutions for expanding features to a higher dimension tthat is different that of the inpjut features
        pepx3 = Conv2D(filter_channels[2], (3, 3), activation='relu', padding='same', name = name+'Depth-wise Representation')(pepx2) #Depth-wise Representation: efficientt 3x3 depth-wise convolutions for learning spatial characteristics to minimize computational complexity while preserving representational capacity
        pepx4 = Conv2D(filter_channels[3], (1, 1), activation='relu', name = name+'Second-stage Projection')(pepx3) #Second-stage Projection: 1x1 convolutions for projecting features back to a lower dimension
        pepx5 = Conv2D(filter_channels[4], (1, 1), activation='relu', name = name+'Extension')(pepx4) #Extension: 1x1 convolutions that finally extend channel dimensinality to a higher dimension to produce the final features
        
        output = Concatenate()([pepx1, pepx2, pepx3, pepx4, pepx5])

        return output
    
    
    input = Input(input_shape)
    x = Conv2D(filters=64, kernel_size=(7, 7), activation='relu', padding='same', strides=(2,2))(input)
    x = MaxPool2D((2, 2))(x)
    
    x11 = PEPX(x, 256, 'PEPX1.1')
    x12 = PEPX(x11, 256, 'PEPX1.2')
    x13 = PEPX(x12, 256, 'PEPX1.3')
    x13 = keras.layers.add([x11, x13])#1
    
    x21 = PEPX(add([x11, x12, x13]), 512, 'PEPX2.1')#2
    x21 = MaxPool2D((2, 2))(x21)
    x22 = PEPX(x21, 512, 'PEPX2.2')
    x23 = PEPX(add([x21, x22]), 512, 'PEPX2.3')#3
    x24 = PEPX(add([x21, x22, x23]), 512, 'PEPX2.4')#4
    
    x31 = PEPX(add([x21, x22, x23, x24]), 1024, 'PEPX3.1')
    x32 = MaxPool2D((2, 2))(x31)
    x32 = PEPX(x31, 1024, 'PEPX3.2')
    x33 = PEPX(add([x31, x32]), 1024, 'PEPX3.3')
    x34 = PEPX(add([x31, x32, x33]), 1024, 'PEPX3.4')
    x35 = PEPX(add([x31, x32, x33, x34]), 1024, 'PEPX3.5')
    x36 = PEPX(add([x31, x32, x33, x34, x35]), 1024, 'PEPX3.6')
    
    x41 = PEPX(add([x31, x32, x33, x34, x35, x36]), 2048, 'PEPX4.1')
    x41 = MaxPool2D(pool_size=(2, 2))(x41)
    x42 = PEPX(x41, 2048, 'PEPX4.2')
    x43 = PEPX(add([x41, x42]), 2048, 'PEPX4.3')
    # FC
    flatten = Flatten()(add([x41, x42, x43]))
    dense = Dense(1024, activation='relu')(flatten)
    dense = Dense(256, activation='relu')(dense)
    output = Dense(4, activation='softmax')(dense)
    
    model = Model(input, output)
    return model


if __name__ == '__main__':
    model = keras_model_build()
    model.summary()

        