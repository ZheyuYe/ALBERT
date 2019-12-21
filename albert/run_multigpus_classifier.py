# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
"""BERT finetuning runner with sentence piece tokenization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from albert import classifier_utils, modeling, tokenization

from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import tpu as contrib_tpu

from tensorflow.python.distribute.cross_device_ops import AllReduceCrossDeviceOps
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.estimator import Estimator
import numpy as np
import time
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

class TimeHistory(tf.estimator.SessionRunHook):
    def __init__(self):
        super(TimeHistory, self).__init__()
        self.times = []
    def begin(self):
        self.times = []
    def before_run(self, run_context):
        self.iter_time_start = time.time()
    def after_run(self, run_context, run_values):
        self.times.append(time.time() - self.iter_time_start)

flags = tf.flags

FLAGS = flags.FLAGS
# set up the dynamic gpu usage
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "albert_config_file", None,
    "The config json file corresponding to the pre-trained ALBERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string(
    "vocab_file", None,
    "The vocabulary file that the ALBERT model was trained on.")

flags.DEFINE_string("spm_model_file", None,
                    "The model file for sentence piece tokenization.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("cached_dir", None,
                    "Path to cached training and dev tfrecord file. "
                    "The file will be generated if not exist.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("train_step", 1000,
                     "Total number of training steps to perform.")

flags.DEFINE_integer(
    "warmup_step", 0,
    "number of steps to perform linear learning rate warmup for.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("keep_checkpoint_max", 5,
                     "How many checkpoints to keep.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("optimizer", "adamw", "Optimizer to use")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

#additional parameters for multi GPU
flags.DEFINE_enum(
    "strategy_type", "mirror", ["one", "mirror"],
    "Training strategy for single or multi gpu training")

flags.DEFINE_integer(
    "num_gpu_cores", 1,
    "Total number of GPU cores to use.")

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "cola": classifier_utils.ColaProcessor,
        "mnli": classifier_utils.MnliProcessor,
        "mrpc": classifier_utils.MrpcProcessor,
        # "xnli": XnliProcessor,
        "sts-b": classifier_utils.StsbProcessor,
        "qqp": classifier_utils.QqpProcessor,
        "sst-2": classifier_utils.Sst2Processor,
        "qnli": classifier_utils.QnliProcessor,
        "rte": classifier_utils.RteProcessor,
        "wnli": classifier_utils.WnliProcessor,
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    albert_config = modeling.AlbertConfig.from_json_file(FLAGS.albert_config_file)

    if FLAGS.max_seq_length > albert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the ALBERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, albert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name](
        use_spm=True if FLAGS.spm_model_file else False,
        do_lower_case=FLAGS.do_lower_case)

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case,
        spm_model_file=FLAGS.spm_model_file)



    # multiple gpus
    NUM_GPUS = FLAGS.num_gpu_cores
    using_customized_optimizer = None
    if NUM_GPUS > 1 and FLAGS.strategy_type == "mirror":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in list(range(NUM_GPUS))])
        strategy = tf.contrib.distribute.MirroredStrategy(
            num_gpus=NUM_GPUS,
            cross_device_ops=AllReduceCrossDeviceOps('nccl', num_packs=NUM_GPUS),
        )
        using_customized_optimizer = True
        tf.logging.info('Use MirroredStrategy with %d devices.',
                        strategy.num_replicas_in_sync)
    else:
        strategy = tf.distribute.OneDeviceStrategy("GPU:0")
        using_customized_optimizer = False
        tf.logging.info('Single device mode.')

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = contrib_tpu.InputPipelineConfig.PER_HOST_V2
    run_config = contrib_tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=contrib_tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host),
        train_distribute=strategy,
        # eval_distribute=strategy, #get error during evaluation
    )


    train_examples = None
    total_time = None

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)

    model_fn = classifier_utils.model_fn_builder(
        albert_config=albert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.train_step,
        num_warmup_steps=FLAGS.warmup_step,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        task_name=task_name,
        customized=using_customized_optimizer,
        optimizer=FLAGS.optimizer)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tf.logging.info("Use TPUEstimator")
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            predict_batch_size=FLAGS.predict_batch_size)
    else:
        tf.logging.info("Use normal Estimator")
        estimator = Estimator(
            model_fn=model_fn,
            params={},
            config=run_config,
        )

    if FLAGS.do_train:
        cached_dir = FLAGS.cached_dir
        if not cached_dir:
          cached_dir = FLAGS.output_dir
        train_file = os.path.join(cached_dir, task_name + "_train.tf_record")
        if not tf.gfile.Exists(train_file):
          classifier_utils.file_based_convert_examples_to_features(
              train_examples, label_list, FLAGS.max_seq_length, tokenizer,
              train_file, task_name)

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info(f"  Batch size = {FLAGS.train_batch_size} * {NUM_GPUS}")
        tf.logging.info("  Num steps = %d", FLAGS.train_step)

        train_input_fn = classifier_utils.file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True,
            task_name=task_name,
            use_tpu=FLAGS.use_tpu,
            bsz=FLAGS.train_batch_size)

        time_hist = TimeHistory()
        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_step, hooks=[time_hist])
        total_time = sum(time_hist.times)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_features = classifier_utils.convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, task_name)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        cached_dir = FLAGS.cached_dir
        if not cached_dir:
          cached_dir = FLAGS.output_dir
        eval_file = os.path.join(cached_dir, task_name + "_eval.tf_record")
        if not tf.gfile.Exists(eval_file):
          classifier_utils.file_based_convert_examples_to_features(
              eval_examples, label_list, FLAGS.max_seq_length, tokenizer,
              eval_file, task_name)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = classifier_utils.file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder,
            task_name=task_name,
            use_tpu=FLAGS.use_tpu,
            bsz=FLAGS.eval_batch_size)

        def _find_valid_cands(curr_step):
          filenames = tf.gfile.ListDirectory(FLAGS.output_dir)
          candidates = []
          for filename in filenames:
            if filename.endswith(".index"):
              ckpt_name = filename[:-6]
              idx = ckpt_name.split("-")[-1]
              if idx != "best" and int(idx) > curr_step:
                candidates.append(filename)
          return candidates

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        checkpoint_path = os.path.join(FLAGS.output_dir, "model.ckpt-best")

        if task_name == "sts-b":
          key_name = "pearson"
        elif task_name == "cola":
          key_name = "matthew_corr"
        else:
          key_name = "eval_accuracy"

        if tf.gfile.Exists(checkpoint_path + ".index"):
          result = estimator.evaluate(
              input_fn=eval_input_fn,
              steps=eval_steps,
              checkpoint_path=checkpoint_path)
          best_perf = result[key_name]
          global_step = result["global_step"]
        else:
          global_step = -1
          best_perf = -1
          checkpoint_path = None
        writer = tf.gfile.GFile(output_eval_file, "w")

        writer.write(f"total time with {NUM_GPUS} GPU(s): {total_time} seconds\n")
        avg_time_per_batch = np.mean(time_hist.times)
        writer.write(f"{FLAGS.train_batch_size * NUM_GPUS / avg_time_per_batch} instances/second with {NUM_GPUS} GPU(s)\n")



        while global_step < FLAGS.train_step:
          steps_and_files = {}
          filenames = tf.gfile.ListDirectory(FLAGS.output_dir)
          for filename in filenames:
            if filename.endswith(".index"):
              ckpt_name = filename[:-6]
              cur_filename = os.path.join(FLAGS.output_dir, ckpt_name)
              if cur_filename.split("-")[-1] == "best":
                continue
              gstep = int(cur_filename.split("-")[-1])
              if gstep not in steps_and_files:
                tf.logging.info("Add {} to eval list.".format(cur_filename))
                steps_and_files[gstep] = cur_filename
          tf.logging.info("found {} files.".format(len(steps_and_files)))
          if not steps_and_files:
            tf.logging.info("found 0 file, global step: {}. Sleeping."
                            .format(global_step))
            time.sleep(1)
          else:
            for checkpoint in sorted(steps_and_files.items()):
              step, checkpoint_path = checkpoint
              if global_step >= step:
                if len(_find_valid_cands(step)) > 1:
                  for ext in ["meta", "data-00000-of-00001", "index"]:
                    src_ckpt = checkpoint_path + ".{}".format(ext)
                    tf.logging.info("removing {}".format(src_ckpt))
                    tf.gfile.Remove(src_ckpt)
                continue
              result = estimator.evaluate(
                  input_fn=eval_input_fn,
                  steps=eval_steps,
                  checkpoint_path=checkpoint_path)
              global_step = result["global_step"]
              tf.logging.info("***** Eval results *****")
              tf.logging.info(f"num_gpu_cores =  {NUM_GPUS}")
              tf.logging.info(f"total_time = {total_time}")
              tf.logging.info(f"speed = {FLAGS.train_batch_size * NUM_GPUS / avg_time_per_batch}")

              for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
              writer.write("best = {}\n".format(best_perf))
              if result[key_name] > best_perf:
                best_perf = result[key_name]
                for ext in ["meta", "data-00000-of-00001", "index"]:
                  src_ckpt = checkpoint_path + ".{}".format(ext)
                  tgt_ckpt = checkpoint_path.rsplit(
                      "-", 1)[0] + "-best.{}".format(ext)
                  tf.logging.info("saving {} to {}".format(src_ckpt, tgt_ckpt))
                  tf.gfile.Copy(src_ckpt, tgt_ckpt, overwrite=True)
                  writer.write("saved {} to {}\n".format(src_ckpt, tgt_ckpt))

              if len(_find_valid_cands(global_step)) > 1:
                for ext in ["meta", "data-00000-of-00001", "index"]:
                  src_ckpt = checkpoint_path + ".{}".format(ext)
                  tf.logging.info("removing {}".format(src_ckpt))
                  tf.gfile.Remove(src_ckpt)
              writer.write("=" * 50 + "\n")
        writer.close()

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")

        classifier_utils.file_based_convert_examples_to_features(
            predict_examples, label_list,
            FLAGS.max_seq_length, tokenizer,
            predict_file, task_name)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = classifier_utils.file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder,
            task_name=task_name,
            use_tpu=FLAGS.use_tpu,
            bsz=FLAGS.predict_batch_size)

        checkpoint_path = os.path.join(FLAGS.output_dir, "model.ckpt-best")
        result = estimator.predict(
            input_fn=predict_input_fn,
            checkpoint_path=checkpoint_path)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        output_submit_file = os.path.join(FLAGS.output_dir, "submit_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as pred_writer,\
            tf.gfile.GFile(output_submit_file, "w") as sub_writer:
          sub_writer.write("index" + "\t" + "prediction\n")
          num_written_lines = 0
          tf.logging.info("***** Predict results *****")
          for (i, (example, prediction)) in\
              enumerate(zip(predict_examples, result)):
            probabilities = prediction["probabilities"]
            if i >= num_actual_predict_examples:
              break
            output_line = "\t".join(
                str(class_probability)
                for class_probability in probabilities) + "\n"
            pred_writer.write(output_line)

            if task_name != "sts-b":
              actual_label = label_list[int(prediction["predictions"])]
            else:
              actual_label = str(prediction["predictions"])
            sub_writer.write(example.guid + "\t" + actual_label + "\n")
            num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("spm_model_file")
    flags.mark_flag_as_required("albert_config_file")

    flags.mark_flag_as_required("init_checkpoint")
    flags.mark_flag_as_required("strategy_type")
    flags.mark_flag_as_required("num_gpu_cores")

    tf.app.run()
