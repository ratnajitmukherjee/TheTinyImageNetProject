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
 " -----------------------------------------------------------------------------
 " Description: Test trained networks using the test set of the TinyImageNet project
 " Author: Ratnajit Mukherjee, ratnajitmukherjee@gmail.com
 " Date: October 2018
"""
from keras.models import load_model
from hdf5datasetgenerator import HDF5DatasetGenerator
from BasicPreprocessor import BasicPreprocessing
from ImagetoArrayPreprocessor import ImagetoArrayPreprocessor
from MeanPreprocessor import MeanPreprocessing
import numpy as np
import json
import os


class EvaluateTinyImageNet:
    def __init__(self, root_path):
        print('\n Evaluating the Tiny ImageNet dataset (test split)')
        self.root_path = root_path

    """
    Function to determine rank 1 and rank 5 accuracy of a classifier network
    """
    def ranked_accuracy(self, preds, labels):
        rank_5 = 0
        rank_1 = 0

        for (pred_label, gt_label) in zip(preds, labels):
            # sort the prediction indexes in a descending order
            pred_label = np.argsort(pred_label, kind='quicksort')[::-1]
            if gt_label in pred_label[:5]:
                rank_5 += 1
            if gt_label in pred_label[:1]:
                rank_1 += 1

        # calculating overall predictions
        rank_5 /= float(len(labels))
        rank_1 /= float(len(labels))

        return rank_1, rank_5

    def evaluate_results(self, pretrained_model_name, num_classes):
        # load the pretrained model
        print('\n [INFO] Loading pre-trained model...')
        model = load_model(os.path.join(self.root_path, pretrained_model_name))

        print('\n [INFO] Loading RGB mean from JSON dump...')
        fid = open(os.path.join(self.root_path, 'rgb_mean.txt'), 'r')
        rgb_mean = json.load(fid)
        fid.close()

        print('\n [INFO] Reading and preprocessing the test set')
        test_HDF5 = os.path.join(self.root_path, 'hdf5Files', 'test.hdf5')
        bp = BasicPreprocessing(64, 64)
        mp = MeanPreprocessing(rgb_mean['RMean'], rgb_mean['GMean'], rgb_mean['BMean'])
        iap = ImagetoArrayPreprocessor()

        testGen = HDF5DatasetGenerator(test_HDF5, 64, preprocessors=[bp, mp, iap], classes=num_classes)

        print('\n [INFO] Generating predictions...')
        predictions = model.predict_generator(testGen.generator(), steps=testGen.numImages//64, max_queue_size=128)

        (rank_1, rank_5) = self.ranked_accuracy(predictions, testGen.db['labels'])

        print('\n Ranked accuracy 1 = {:0.2f}'.format(rank_1*100))
        print('\n Ranked accuracy 5 = {:0.2f}'.format(rank_5 * 100))

        return


if __name__ == '__main__':
    root_path = input('Please enter the root path: ')
    evTimgNet = EvaluateTinyImageNet(root_path=root_path)
    evTimgNet.evaluate_results('checkpoint_112-0.44.hdf5', 200)