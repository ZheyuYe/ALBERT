import tensorflow.compat.v1 as tf
import json
tf.enable_eager_execution()

d = tf.data.TFRecordDataset('./cached_albert_tfrecord/SQuAD1.1_train_384.tf_record')

num_samples = 0

l = []
for ele in d:
    num_samples += 1
    l.append(tf.train.Example.FromString(ele.numpy()))
    # print(tf.train.Example.FromString(ele.numpy()))
print(num_samples)  # 88786
# json.dump(l, open('train_feature_file.json', 'w', encoding='utf-8'))
