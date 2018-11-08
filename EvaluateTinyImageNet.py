"""
 " License:
 " -----------------------------------------------------------------------------
 " Copyright (c) 2018, Ratnajit Mukherjee.
 " All rights reserved.
 "
 " Redistribution and use in source and binary forms, with or without
 " modification, are permitted provided that the following conditions are met:
 "
 " 1. Redistributions of source code must retain the above copyright notice,
 "    this list of conditions and the following disclaimer.
 "
 " 2. Redistributions in binary form must reproduce the above copyright notice,
 "    this list of conditions and the following disclaimer in the documentation
 "    and/or other materials provided with the distribution.
 "
 " 3. Neither the name of the copyright holder nor the names of its contributors
 "    may be used to endorse or promote products derived from this software
 "    without specific prior written permission.
 "
 " THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 " AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 " IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 " ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 " LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 " CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 " SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 " INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 " CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 " ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 " POSSIBILITY OF SUCH DAMAGE.
 "
 " -----------------------------------------------------------------------------
 " Description: Evaluate Trained Networks and predict labels on the TinyImageNet
 " Test Set. Note that this test set is NOT the same as created in the test.HDF5.
 " Author: Ratnajit Mukherjee, ratnajitmukherjee@gmail.com
 " Date: November 2018
 " -----------------------------------------------------------------------------
"""
import json
import os

import cv2
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from Preprocessing.BasicPreprocessor import BasicPreprocessing
from Preprocessing.ImagetoArrayPreprocessor import ImagetoArrayPreprocessor
from Preprocessing.MeanPreprocessor import MeanPreprocessing


class PredictTinyImageNet:
    def __init__(self, root_path):
        print('[INFO] Loading the TinyImageNet TEST set.')
        self.root_path = root_path
        self.test_set_path = os.path.join(self.root_path, 'test')
        self.test_images_path = os.path.join(self.test_set_path, 'images')

    def preprocess_image(self, image, rgb_mean):
        # basic pre-processing
        bp = BasicPreprocessing(64, 64)
        image = bp.preprocess(image=image)

        # mean pre-processing
        mp = MeanPreprocessing(rMean=rgb_mean['RMean'], gMean=rgb_mean['GMean'], bMean=rgb_mean['BMean'])
        image = mp.preprocess(image=image)

        # image to array
        iap = ImagetoArrayPreprocessor()
        keras_image = iap.preprocess(image=image)   # the last processing is to turn image into a keras array
        keras_image = np.expand_dims(keras_image, axis=0)
        return keras_image

    def predict_class_labels(self, model_name):
        # get the labels from the validation annotations
        print('[INFO] Fetching all the class labels')
        label_annotation_path = os.path.join(self.root_path, 'val', 'val_annotations.txt')
        label_contents = open(label_annotation_path).read().strip().split('\n')
        label_contents = [line.split('\t')[:2] for line in label_contents]
        labels = [line[1] for line in label_contents]

        # fit transform the label encoder so that we can inverse the transform to predict images
        le = LabelEncoder()
        labels = le.fit_transform(labels)   # this gives the full 10K labels but more importantly gives you the fit

        # now we list out of the files in the TinyImageNet test set
        print('[INFO] Fetching list of prediction images')
        test_list = [filename for filename in os.listdir(self.test_images_path) if filename.endswith('.JPEG')]

        # now we load the TinyImageNet image mean
        print('[INFO] Obtaining RGB-MEAN of the TinyImageNet dataset')
        fid = open(os.path.join(self.root_path, 'rgb_mean.txt'), 'r')
        rgb_mean = json.load(fid)

        # now we load the pre-trained model (neural network)
        print('[INFO] Loading the pre-trained model')
        model = load_model(os.path.join(self.root_path, model_name))

        # now we load each image and predict the label
        print('[INFO] Starting prediction and writing to prediction file..')
        fid = open(os.path.join(self.test_set_path, 'predictions.txt'), 'w')

        for (i, filename) in zip(tqdm(range(int(len(test_list)))), test_list):
            file_abs_path = os.path.join(os.path.join(self.test_images_path, filename))
            img = cv2.imread(filename=file_abs_path)
            k_image = self.preprocess_image(image=img, rgb_mean=rgb_mean)
            pred_class = np.argmax(model.predict(k_image))
            pred_label = le.inverse_transform(pred_class)
            fid.write('{0} {1}\n'.format(filename, pred_label))

        fid.close()
        print('[INFO] Prediction Complete and file closed...')


if __name__ == '__main__':
    root_path = input('Please enter the root path: ')
    model_name = 'TinyImageNet_InceptionV4_Acc-0.46.hdf5'
    pred = PredictTinyImageNet(root_path=root_path)
    pred.predict_class_labels(model_name=model_name)