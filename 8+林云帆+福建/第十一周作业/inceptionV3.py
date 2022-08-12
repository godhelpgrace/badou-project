from keras import backend as K
from keras.layers import Conv2D,BatchNormalization,Activation,Input,MaxPooling2D,AveragePooling2D,GlobalAveragePooling2D,Dense
K.set_image_data_format('channels_last')
from keras import layers
from  keras import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
import numpy as np
def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              strides = (1,1),
              padding = 'same',
              name = None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(filters, (num_row,num_col), strides=strides,padding=padding,use_bias=False,
               name = conv_name)(x)
    x = BatchNormalization(scale=False, name = bn_name)(x)
    x = Activation('relu',name=name)(x)

    return x

def InceptionV3(input_shape=[299,299,3],
                classes = 1000):
    img_input = Input(shape = input_shape)

    x = conv2d_bn(img_input,32,3,3,strides=(2,2),padding='valid')
    x = conv2d_bn(x,32,3,3,padding='valid')
    x = conv2d_bn(x,64,3,3)
    x = MaxPooling2D((3,3),strides=(2,2))(x)

    x = conv2d_bn(x,80,1,1,padding='valid')
    x = conv2d_bn(x,192,3,3,padding='valid')
    x = MaxPooling2D((3,3),strides=(2,2))(x)

    #Block1 part1
    branch1_1 = conv2d_bn(x,64,1,1)

    branch5_5 = conv2d_bn(x,48,1,1)
    branch5_5 = conv2d_bn(branch5_5,64,5,5)

    branch3_3 = conv2d_bn(x,64,1,1)
    branch3_3 = conv2d_bn(branch3_3,96,3,3)
    branch3_3 = conv2d_bn(branch3_3,96,3,3)

    branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
    branch_pool = conv2d_bn(branch_pool,32,1,1)

    x = layers.concatenate([branch1_1,branch5_5,branch3_3,branch_pool],axis=3,name='mixed0')
    #Block1 part2
    branch1_1 = conv2d_bn(x,64,1,1)

    branch5_5 = conv2d_bn(x,48,1,1)
    branch5_5 = conv2d_bn(branch5_5,64,5,5)

    branch3_3 = conv2d_bn(x,64,1,1)
    branch3_3 = conv2d_bn(branch3_3,96,3,3)
    branch3_3 = conv2d_bn(branch3_3,96,3,3)

    branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
    branch_pool = conv2d_bn(branch_pool,64,1,1)

    x = layers.concatenate([branch1_1,branch5_5,branch3_3,branch_pool],axis=3,name = 'mixed1')

    #Block1 part3
    branch1_1 = conv2d_bn(x,64,1,1)

    branch5_5 = conv2d_bn(x,48,1,1)
    branch5_5 = conv2d_bn(branch5_5,64,5,5)

    branch3_3 = conv2d_bn(x,64,1,1)
    branch3_3 = conv2d_bn(branch3_3,96,3,3)
    branch3_3 = conv2d_bn(branch3_3,96,3,3)

    branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
    branch_pool = conv2d_bn(branch_pool,64,1,1)

    x = layers.concatenate([branch1_1,branch5_5,branch3_3,branch_pool],axis=3,name='mixed2')

    #Block2 part1
    branch3_3 = conv2d_bn(x, 384,3,3,strides=(2,2),padding = 'valid')

    branch3_3dbl = conv2d_bn(x,64,1,1)
    branch3_3dbl = conv2d_bn(branch3_3dbl,96,3,3)
    branch3_3dbl = conv2d_bn(branch3_3dbl,96,3,3,strides=(2,2),padding='valid')

    branch_pool = MaxPooling2D((3,3),strides=(2,2))(x)
    x = layers.concatenate([branch3_3,branch3_3dbl,branch_pool],axis=3,name ='mixed3')

    #Block2 part2
    branch1_1 = conv2d_bn(x,192,1,1)

    branch7_7 = conv2d_bn(x,128,1,1)
    branch7_7 = conv2d_bn(branch7_7,128,1,7)
    branch7_7 = conv2d_bn(branch7_7,192,7,1)

    branch7_7dbl = conv2d_bn(x,128,1,1)
    branch7_7dbl = conv2d_bn(branch7_7dbl,128,7,1)
    branch7_7dbl = conv2d_bn(branch7_7dbl,128,1,7)
    branch7_7dbl = conv2d_bn(branch7_7dbl,128,7,1)
    branch7_7dbl = conv2d_bn(branch7_7dbl,192,1,7)

    branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
    branch_pool =conv2d_bn(branch_pool,192,1,1)

    x = layers.concatenate([branch1_1, branch7_7, branch7_7dbl, branch_pool],axis=3,name = 'mixed4')

    #Block2 part3,4
    for i in range(2):
        branch1_1 = conv2d_bn(x,192,1,1)

        branch7_7 = conv2d_bn(x,160,1,1)
        branch7_7 = conv2d_bn(branch7_7,160,1,7)
        branch7_7 = conv2d_bn(branch7_7,192,7,1)

        branch7_7dbl = conv2d_bn(x,160,1,1)
        branch7_7dbl = conv2d_bn(branch7_7dbl,160,7,1)
        branch7_7dbl = conv2d_bn(branch7_7dbl,160,1,7)
        branch7_7dbl = conv2d_bn(branch7_7dbl,160,7,1)
        branch7_7dbl = conv2d_bn(branch7_7dbl,192,1,7)

        branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
        branch_pool = conv2d_bn(branch_pool,192,1,1)
        x = layers.concatenate([branch1_1,branch7_7,branch7_7dbl,branch_pool],axis=3,name = 'mixed'+str(5+i))

    #Block part5
    branch1_1 = conv2d_bn(x,192,1,1)

    branch7_7 = conv2d_bn(x,192,1,1)
    branch7_7 = conv2d_bn(branch7_7,192,1,7)
    branch7_7 = conv2d_bn(branch7_7,192,7,1)

    branch7_7dbl = conv2d_bn(x,192,1,1)
    branch7_7dbl = conv2d_bn(branch7_7dbl,192,7,1)
    branch7_7dbl = conv2d_bn(branch7_7dbl,192,1,7)
    branch7_7dbl = conv2d_bn(branch7_7dbl,192,7,1)
    branch7_7dbl = conv2d_bn(branch7_7dbl,192,1,7)

    branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
    branch_pool = conv2d_bn(branch_pool,192,1,1)
    x = layers.concatenate([branch1_1,branch7_7,branch7_7dbl,branch_pool],axis=3,name='mixed7')

    #Block3 part1
    branch3_3 =conv2d_bn(x,192,1,1)
    branch3_3 =conv2d_bn(branch3_3,320,3,3,strides=(2,2),padding='valid')

    branch7_7_3 = conv2d_bn(x,192,1,1)
    branch7_7_3 = conv2d_bn(branch7_7_3,192,1,7)
    branch7_7_3 = conv2d_bn(branch7_7_3,192,7,1)
    branch7_7_3 = conv2d_bn(branch7_7_3,192,3,3,strides=(2, 2), padding='valid')

    branch_pool =MaxPooling2D((3,3),strides=(2,2),padding='valid')(x)

    x =layers.concatenate([branch3_3,branch7_7_3,branch_pool],axis=3,name='mixed8')

    #Block3 part2,3
    for i in range(2):
        branch1_1 = conv2d_bn(x,320,1,1)

        branch3_3 = conv2d_bn(x, 384,1,1)
        branch3_3_1 = conv2d_bn(branch3_3,384,1,3)
        branch3_3_2 = conv2d_bn(branch3_3,384,3,1)
        branch3_3  =layers.concatenate([branch3_3_1,branch3_3_2],axis =3,name='mixed9_'+str(i))

        branch3_3dbl =conv2d_bn(x,448,1,1)
        branch3_3dbl =conv2d_bn(branch3_3dbl,384,3,3)
        branch3_3dbl_1 =conv2d_bn(branch3_3dbl,384,1,3)
        branch3_3dbl_2 =conv2d_bn(branch3_3dbl,384,3,1)
        branch3_3dbl =layers.concatenate([branch3_3dbl_1,branch3_3dbl_2],axis=3)

        branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
        branch_pool =conv2d_bn(branch_pool,192,1,1)
        x =layers.concatenate([branch1_1,branch3_3,branch3_3dbl,branch_pool],axis = 3,name = 'mixed' +str(i+9))

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes,activation='softmax',name = 'predictions')(x)

    inputs = img_input
    model = Model(inputs,x,name = 'inception_v3')

    return model

def preprocess_input(x):
    x /= 255.
    x -=0.5
    x *=2.
    return x

if __name__ == '__main__':
    model =InceptionV3()
    model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")

    img_path = 'elephant.jpg'
    img = image.load_img(img_path,target_size=(299,299))
    img = image.img_to_array(img)
    #print(img.shape)
    x = np.expand_dims(img,axis=0)
    #print(x.shape)
    x = preprocess_input(x)
    preds = model.predict(x)
    print('Predicted:',decode_predictions(preds))