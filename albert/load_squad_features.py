import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

d = tf.TFRecordDataset('albert_base_v2_squad_1.1_finetune/train_feature_file.fea')

for ele in d:
    print(tf.train.Example.FromString(ele.numpy()))
