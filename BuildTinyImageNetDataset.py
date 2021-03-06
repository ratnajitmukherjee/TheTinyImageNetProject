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
 "
 " Author: Ratnajit Mukherjee, ratnajitmukherjee@gmail.com
 " Date: October 2018
"""
import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from hdf5io.hdf5datasetwriter import HDF5DatasetWriter


class BuildTinyImageNetDataset:
    def __init__(self, root_path):
        self.global_train_path = os.path.join(root_path, 'train')
        self.global_val_path = os.path.join(root_path, 'val', 'images')
        self.global_output_path = os.path.join(root_path, 'hdf5Files')
        self.val_mappings = os.path.join(root_path, 'val', 'val_annotations.txt')

        # The wordnet IDs are used to search the words in the words txt file and thus join
        # and create the data labels
        self.global_wordnet_id = os.path.join(root_path, 'wnids.txt')
        self.global_words = os.path.join(root_path, 'words.txt')
        print("\n Starting to build TinyImageProject dataset for image classification...")

    def configDataSet(self):
        if not os.path.exists(self.global_output_path):
            print('\n HDF5 output directory does not exist. Creating a new directory')
            os.makedirs(self.global_output_path)
        train_HDF5 = os.path.join(self.global_output_path, 'train.hdf5')
        val_HDF5 = os.path.join(self.global_output_path, 'val.hdf5')
        test_HDF5 = os.path.join(self.global_output_path, 'test.hdf5')
        return train_HDF5, val_HDF5, test_HDF5

    def buildDataSet(self):
        # safety check to test whether files have already been built or not
        # extract all the training paths from the subdirs

        train_paths = [os.path.join(root, filename) for root, subdirs, files in os.walk(self.global_train_path)
                       for filename in files if filename.endswith(".JPEG")]

        train_labels = [filepath.split(os.path.sep)[-3] for filepath in train_paths]

        # convert training labels to unique integer values
        le = LabelEncoder()
        train_labels = le.fit_transform(train_labels)

        # In TinyImageNet project, we don't have access to test data. Therefore, we split train data -> 10% for test
        (train_paths, test_paths, train_labels, test_labels) = train_test_split(train_paths, train_labels,
                                                                                test_size=0.1, stratify=train_labels,
                                                                                random_state=20)

        # Next we handle the validation paths creating the validation labels
        val_contents = open(self.val_mappings).read().strip().split('\n')
        val_contents = [line.split('\t')[:2] for line in val_contents]
        val_paths = [os.path.join(self.global_val_path, line[0]) for line in val_contents]
        val_labels = le.fit_transform([line[1] for line in val_contents])

        # Now we have train, val and test paths and labels. Next building the datasets
        (train_HDF5, val_HDF5, test_HDF5) = self.configDataSet()

        train_val_test_dataset = [('train', train_paths, train_labels, train_HDF5),
                                  ('val', val_paths, val_labels, val_HDF5),
                                  ('test', test_paths, test_labels, test_HDF5)]

        (RList, GList, BList) = ([], [], [])

        for (usage, paths, labels, output_path) in train_val_test_dataset:
            print('\n Building dataset {0}...'.format(output_path))
            dswriter = HDF5DatasetWriter((len(paths), 64, 64, 3), outputPath=output_path)
            for (i, path, label) in zip(tqdm(range(int(len(paths)))), paths, labels):
                img = cv2.imread(path)  # the image is read in BGR and not RGB order
                if usage == 'train':
                    RList.append(np.mean(np.ravel(img[:, :, 2])))
                    GList.append(np.mean(np.ravel(img[:, :, 1])))
                    BList.append(np.mean(np.ravel(img[:, :, 0])))
                dswriter.add([img], [label])
            dswriter.close()
            print('\n Finished building dataset {0}'.format(output_path))

        print('[PROGRESS INFO: ] Extracting training data mean.')
        rgb_mean = {'RMean': np.mean(RList), 'GMean': np.mean(GList), 'BMean': np.mean(BList)}
        return rgb_mean


if __name__ == '__main__':
    root_path = input("\n Please enter the root path: ")
    buildTinyImageNet = BuildTinyImageNetDataset(root_path)
    rgb_mean = buildTinyImageNet.buildDataSet()
