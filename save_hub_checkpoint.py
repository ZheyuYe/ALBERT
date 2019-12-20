import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import os

def save_model(model_size,version):
    url = "https://tfhub.dev/google/albert_{}/{}".format(model_size.lower(), version)
    print(url)
    model = hub.Module(url, trainable=False)
    # print(url)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        ckpt_file_path=os.path.abspath(model_size+'_v'+str(version))
        print(ckpt_file_path)
        print(os.getcwd())
        if os.path.isdir(ckpt_file_path) is False:
            os.makedirs(ckpt_file_path)
        tf.train.Saver().save(sess, ckpt_file_path+'/model.ckpt')

for ALBERT_MODEL in ['base','large','xlarge','xxlarge']:
# for ALBERT_MODEL in ['base']:
    model_size = ALBERT_MODEL.lower()
    save_model(model_size,1)
    save_model(model_size,2)
