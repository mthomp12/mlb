# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 09:37:54 2020

@author: mthom
"""

import warnings
warnings.simplefilter('ignore')

import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers import Dense, Flatten, Conv1D
from keras.models import Sequential
from CNN_data_generator import DataGenerator
import keras
import keras.backend as K

#Set learning rate
myadam = keras.optimizers.Adam(learning_rate=0.001) # default is 0.001
#shape0, shape1 = (78, 500)
shape0, shape1 = (107, 500)
batch_size = 25
epochs = 1
steps_per_epoch = False #False to run thru entire batch per Epoch, else #records per epoch = steps_per_epoch * batch_size
save_predictions = False
limit = False
show_train_results = False


def correlation(x, y):    
    mx = tf.math.reduce_mean(x)
    my = tf.math.reduce_mean(y)
    xm, ym = x-mx, y-my
    r_num = tf.math.reduce_mean(tf.multiply(xm,ym))        
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    return r_num / r_den

def corr_loss(x,y):
    return 1 - correlation(x, y)


def combined_loss(x, y):
    return K.square(x - y) / 100 * corr_loss(x,y)**0.5


    
def err(arr1, arr2):
    arr1, arr2 = arr1.reshape(-1), arr2.reshape(-1)
    mae = np.mean(abs(arr1 - arr2))
    mse = np.mean(abs(arr1 - arr2)**2)**0.5
    corr = np.corrcoef(arr1,arr2)[0][1]
    r2 = np.sum((arr2-arr1.mean())**2)/np.sum((arr1-arr1.mean())**2)
    return mae, mse, corr, r2
 

params = {'dim': (shape1, shape0+1),
          'batch_size': batch_size,
          'shuffle': True,
          'transpose1D': True, 
          'save_y': False,
          'limit': limit,
          #'cols':range(20)
          }

params_test = params.copy()
params_test['shuffle'] = False

training_generator = DataGenerator('F:/NN/data/train_x', **params)
test_generator = DataGenerator('F:/NN/data/test_x', **params_test)

# Design model
model = Sequential()
model.add(Conv1D(input_shape=(shape0, shape1),filters=64, kernel_size=(1), activation='relu'))
model.add(Conv1D(filters=64, kernel_size=(1), activation='relu'))
model.add(Flatten())
model.add(Dense(units=64, activation = 'relu'))
model.add(Dense(units=32, activation = 'relu'))
model.add(Dense(units=16, activation = 'relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer = myadam, loss = 'mse', metrics=['mae',correlation])
#model.compile(optimizer = myadam, loss = combined_loss, metrics=['mse','mae',correlation])
print(model.summary())

if steps_per_epoch:
    history = model.fit_generator(generator=training_generator, epochs=epochs, steps_per_epoch=steps_per_epoch)
else:
    history = model.fit_generator(generator=training_generator, epochs=epochs)    



test_generator.save_y = True
y_hat_test = model.predict_generator(generator=test_generator)
test_generator.save_y = False
print('\ntest results:')

y_test = test_generator.y
mae2, rmse2, corr2, r2 = err(y_test, y_hat_test)
res = pd.DataFrame({'mae':[mae2], 'rmse':[rmse2],'corr':[corr2]}, index=['test'])
res['r^2'] = res['corr']**2
print(res, '\n\n')



if show_train_results:
    print('Calculating training set results...')
    train_results = model.evaluate(training_generator)
    train_results[0] = train_results[0]**0.5
    train_results += [train_results[2]**2]
    train_results = pd.DataFrame(train_results, index=['rmse','mae','corr','r^2'], columns=['train']).T
    train_results = train_results[['mae','rmse','corr','r^2']] 
    print(train_results)

if save_predictions:
    if test_generator.shuffle:
        raise Exception('This code only works if data is not shuffled!')
    file = test_generator.files[0]
    f_start = file.find('x_')
    files = [file[f_start:-4] for file in test_generator.files]
    lim = (len(files)%batch_size)
    if lim:
        files = files[:-lim]
    df = pd.DataFrame({'file':files,'y_hat':y_hat_test.reshape(-1),'y':y_test})    
    df.to_pickle('predictions.pkl')
