import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, merge, Merge
from keras.layers import Concatenate, Flatten
from keras.optimizers import SGD
from keras.layers import LSTM, Input
from keras.models import Model, optimizers
from keras.models import load_model

from keras import regularizers
import numpy as np

def buildModel(sizeLen,sizeVectors):
    forwardS=Sequential()
    forwardS.add(LSTM(units=sizeVectors, input_shape=(sizeLen,sizeVectors), return_sequences=False))

    backwardS=Sequential()
    backwardS.add(LSTM(units=sizeVectors, input_shape=(sizeLen,sizeVectors), return_sequences=False))
    
    extraInfoS=Sequential()

    extraInfoS.add(Dense(input_dim=sizeVectors, units=sizeVectors, activation=None))
    
    merged=Merge([backwardS,forwardS,extraInfoS],mode='concat')

    finalModel=Sequential()
    finalModel.add(merged)
    finalModel.add(Dense((sizeVectors*3), activation='softplus'))
    finalModel.add(Dense((sizeVectors*2),activation='softplus'))
    finalModel.add(Dense(sizeVectors, activation='softplus'))
    finalModel.add(Dense(sizeVectors, activation='softplus'))    
    
    if(sizeVectors==100):
        learningRate=0.0005
    elif(sizeVectors==300):
        learningRate=0.0010
    elif(sizeVectors==500):
        learningRate=0.0015
    
    adam1=optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    finalModel.compile(loss='mean_squared_error', optimizer=adam1)#categorical_crossentropy #poisson #logcosh
    return finalModel


def fitModel(trainingData, finalModel, batchSize, epochsFit):
    finalModel.fit([trainingData[0],trainingData[1],trainingData[2]], trainingData[3], epochs=epochsFit, batch_size=batchSize, verbose=2)
