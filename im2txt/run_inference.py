# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary


class Im2txtInference:
    def __init__(self,checkpoint_path,vocab_file):
        self.vocab_file=vocab_file
        self.checkpoint_path=checkpoint_path
        g = tf.Graph()
        with g.as_default():
            self.model = inference_wrapper.InferenceWrapper()
            restore_fn = self.model.build_graph_from_config(configuration.ModelConfig(),
                                                       self.checkpoint_path)
        g.finalize()
        self.vocab = vocabulary.Vocabulary(self.vocab_file)
        self.sess=tf.Session(graph=g)
        restore_fn(self.sess)
        tf.logging.info('Tensorflow Session initialized for im2txt module.')

    def get_image_caption(self,audio_path):
        generator = caption_generator.CaptionGenerator(self.model, self.vocab)
        with tf.gfile.GFile(audio_path, "rb") as f:
            image = f.read()
        captions = generator.beam_search(self.sess, image)
        return captions

    def get_raw_image_caption(self,raw_data):
        generator = caption_generator.CaptionGenerator(self.model, self.vocab)
        captions = generator.beam_search(self.sess, raw_data)
        return captions

