from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import inception_resnet_v2

InceptionResNetV2 = inception_resnet_v2.InceptionResNetV2
decode_predictions = inception_resnet_v2.decode_predictions
preprocess_input = inception_resnet_v2.preprocess_input
