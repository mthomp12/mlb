# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 16:15:37 2020

@author: mthom
"""



import numpy as np
import glob
from tensorflow.python.keras.utils.data_utils import Sequence

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, path, batch_size=2, dim=(500,73), n_channels=1,shuffle=True, steps_per_epoch=False, transpose=False, transpose1D=False, save_y=False, limit=False):
        'Initialization'
        self.limit = limit
        self.transpose = transpose
        self.transpose1D = transpose1D
        self.dim = dim
        self.path = path
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.files = self.get_info()
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.save_y = save_y
        self.y = np.array([])

    def get_info(self):
        #files = glob.glob('./' + self.path + '/*.npy')
        if self.path[-3:]=='all':
            files = glob.glob(self.path[:-3] + 'train2/*.npy') + glob.glob(self.path[:-3] + 'cv2/*.npy')
        else:
            files = glob.glob(self.path + '/*.npy')
        #print(files[0:5])
        if self.limit:
            files = files[:self.limit]
        return files
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.steps_per_epoch:
            return self.steps_per_epoch
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.files[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load(ID, allow_pickle=True)
        
        if self.save_y:
            self.y = np.append(self.y, X[:,0,-1])
            
        if self.transpose:
            xT = np.zeros((self.batch_size,1,X[0,].shape[1]-1,X[0,].shape[0]))
            for i in range(self.batch_size):
                xT[i,0,:,:] = X[i,:,:-1].T
            return xT, np.array(X[:,0,-1])
        
        if self.transpose1D:
            xT = np.zeros((self.batch_size,X[0,].shape[1]-1,X[0,].shape[0]))
            for i in range(self.batch_size):
                xT[i,:,:] = X[i,:,:-1].T
            return xT, np.array(X[:,0,-1])        
        
        return X[:,:,:-1], np.array(X[:,0,-1])
