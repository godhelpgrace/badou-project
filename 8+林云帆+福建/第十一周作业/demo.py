import tensorflow as tf
import utils
from nets.vgg16 import vgg_16



img = utils.load_image('./test_data/dog.jpg')


inputs = tf.placeholder(tf.float32, [None, None , 3])
resize_img = utils.resize_image(inputs, (224,224))

prediction = vgg_16(resize_img)

sess = tf.Session()
ckpt_name = './model/vgg_16.ckpt'
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, ckpt_name)

pro =tf.nn.softmax(prediction)
pre =sess.run(pro,feed_dict={inputs:img})

print("result: ")
utils.print_prob(pre[0],'./synset.txt')