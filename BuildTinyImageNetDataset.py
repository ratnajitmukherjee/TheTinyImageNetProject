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
 " Integrated into the TinyImagenet project by:
 " Author: Ratnajit Mukherjee, ratnajitmukherjee@gmail.com
 " Date: October 2018
"""
import os
import numpy as np
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class BuildTinyImageNetDataset:
    def __init__(self, root_path):
        global_train_path = os.path.join(root_path, 'train')
        global_val_path = os.path.join(root_path, 'val/images')
        global_output_path = os.path.join(root_path, 'hdf5Files')        
        val_mappings = os.path.join(root_path, 'val/val_annotations.txt')

        # The wordnet IDs are used to search the words in the words txt file and thus join
        # and create the data labels
        global_wordnet_id = os.path.join(root_path, 'wnids.txt')
        global_words = os.path.join(root_path, 'words.txt')        
        print("\n Starting to build TinyImageProject dataset for image classification...")

        # uncomment the following lines for testing the paths correctly        
        print("\n Training Path: {0} \n Validation Path: {1} \n Output Path: {2}".format(global_train_path, global_val_path,global_output_path))
        return global_train_path, global_val_path, global_output_path

    def buildDataSet(self, root_path, train_path, val_path, output_path):
        # extract all the training paths from the subdirs
        train_paths = [os.path.join(root, filename) for root, subdirs, files in os.walk(training_image_path) for filename in files if filename.endswith(".JPEG")]
        train_labels = [filepath.split(os.path.sep)[-3] for filepath in train_paths]

        # convert training labels to unique integer values
        le = LabelEncoder()
        train_labels = le.fit_transform(train_labels)

        """
        Since the TinyImageNet dataset does not have any test data set, we split the training dataset into the same number of images as the validation set
        and then use the test set to test our models. That amounts to 0.1 or 10% of the training dataset
        NOTE: Please read the README.md for explanation details
        """
        (train_paths, test_paths, train_labels, test_labels) = train_test_split(train_paths, train_labels, test_size=0.1, stratify=train_labels, random_state=20)
                
        


        


if __name__== '__main__':
    root_path = input("\n Please enter the root path: ")
    buildTinyImageNet = BuildTinyImageNetDataset(root_path)
