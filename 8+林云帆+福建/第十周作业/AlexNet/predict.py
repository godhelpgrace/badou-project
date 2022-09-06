from keras import backend as K
from model.AlexNet_1 import AlexNet
import numpy as np
import  cv2
import utils
K.set_image_data_format('channels_last')

if __name__ == "__main__":
    model = AlexNet()
    img = cv2.imread('./Test.jpg')
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_nor =  img_RGB / 255
    img_nor = np.expand_dims(img_nor,axis=0)
    img_resize = utils.resize_image(img_nor,(224,224))
    model.load_weights('./logs/last3.h5')
    utils.print_answer(np.argmax(model.predict(img_resize)))
    cv2.imshow('000',img)
    cv2.waitKey(0)