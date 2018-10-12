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
 "
 " Author: Ratnajit Mukherjee, ratnajitmukherjee@gmail.com
 " Date: July 2018
"""

"""
Description: Class to preprocess the image according to the requirements of the project. 
1) Simple Preprocessing: resize to the required size if the image file is not in the 
correct input layer size

2) Mean Preprocessing: Subtract the mean of the dataset from the input image

3) Convert the image to Keras array for processing
"""
import cv2
from keras.preprocessing.image import img_to_array

class ImageProcessor:
    def __init__(self, width, height, RMean, GMean, BMean, dataFormat=None):
        self.width = width
        self.height = height
        self.rMean = RMean
        self.gMean = GMean
        self.bMean = BMean
        self.dataFormat = dataFormat

    def preprocess(self, image):
        # simple preprocessing
        img = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)

        # subtraction of mean
        (B, G, R) = cv2.split(img.astype('float32'))
        B -= self.bMean
        G -= self.gMean
        R -= self.rMean
        img = cv2.merge([B, G, R])

        # converting img to keras array
        return img_to_array(img, data_format=self.dataFormat)