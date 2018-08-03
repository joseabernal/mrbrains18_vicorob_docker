import numpy as np

from keras import backend as K
from keras.layers import Activation, Input, PReLU, Flatten, Dense, Cropping3D, Dropout
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers.convolutional import Conv3DTranspose as Deconv3D
from keras.layers.core import Permute, Reshape
from keras.layers.merge import add, concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model

K.set_image_dim_ordering('th')

def generate_combined_model(input_shape, output_shape, num_classes=9, scale=1) :
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    
    model_a = generate_uresnet_model(input_shape, output_shape, num_classes, scale)
    model_b = generate_uresnet_model(input_shape, output_shape, num_classes, scale)
    
    output_a = model_a(input_a)
    output_b = model_b(input_b)
    
    pred = add([output_a, output_b])
    pred = Activation('softmax', name='prediction')(pred)
    
    return Model(inputs=[input_a, input_b], outputs=[output_a, output_b, pred])
    
def generate_uresnet_model(input_shape, output_shape, num_classes=9, scale=1):
    input = Input(shape=input_shape)

    conv1 = get_res_conv_core(input, np.int32(scale*32))
    pool1 = get_pooling(conv1, np.int32(scale*32))

    conv2 = get_res_conv_core(pool1, np.int32(scale*64))
    pool2 = get_pooling(conv2, np.int32(scale*64))

    conv3 = get_res_conv_core(pool2, np.int32(scale*128))
    pool3 = get_pooling(conv3, np.int32(scale*128))

    conv4 = get_res_conv_core(pool3, np.int32(scale*256))
    
    up1 = get_deconv_layer(conv4, np.int32(scale*128))
    conv5 = get_res_conv_core(up1, np.int32(scale*128))

    add35 = merge_add(conv3, conv5)
    conv6 = get_res_conv_core(add35, np.int32(scale*128))
    up2 = get_deconv_layer(conv6, np.int32(scale*64))

    add22 = merge_add(conv2, up2)
    conv7 = get_res_conv_core(add22, np.int32(scale*64))
    up3 = get_deconv_layer(conv7, np.int32(scale*32))

    add13 = merge_add(conv1, up3)
    conv8 = get_res_conv_core(add13, np.int32(scale*32))

    pred = get_conv_fc(conv8, np.int32(num_classes))
    pred = organise_output(pred, output_shape)

    return Model(inputs=[input], outputs=[pred])

def merge_add(a, b) :
    c = add([a, b])
    c = BatchNormalization(axis=1)(c)
    return PReLU()(c)

def get_res_conv_core(input, num_filters) :
    a = Conv3D(num_filters, kernel_size=(3, 3, 3), padding='same')(input)
    b = Conv3D(num_filters, kernel_size=(1, 1, 1), padding='same')(input)
    return merge_add(a, b)

def get_pooling(input, num_filters) :
    a = MaxPooling3D(pool_size=(2, 2, 2))(input)
    a = BatchNormalization(axis=1)(a)
    return PReLU()(a)

def get_deconv_layer(input, num_filters) :
    a = Deconv3D(num_filters, kernel_size=(2, 2, 2), strides=(2, 2, 2))(input)
    a = BatchNormalization(axis=1)(a)
    return PReLU()(a)

def get_conv_fc(input, num_filters=4) :
    fc = Conv3D(num_filters, kernel_size=(1, 1, 1))(input)

    return PReLU()(fc)

def organise_output(input, output_shape) :
    pred = Reshape(output_shape)(input)
    pred = Permute((2, 1))(pred)
    return Activation('softmax')(pred)

def general_loss(y_true, y_pred) :
    return K.categorical_crossentropy(y_true, y_pred)