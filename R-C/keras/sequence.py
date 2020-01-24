import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, merge, Merge
from keras.layers import Concatenate, Flatten, Dropout
from keras.optimizers import SGD
from keras.layers import LSTM, Input
from keras.models import Model, optimizers
from keras.models import load_model

from keras import regularizers
import numpy as np
from keras.layers import GRU

#merged = Concatenate(axis=-1)([forward.output, backward.output])#merge([forward, backward], mode='concat', concat_axis=-1)
#merged = merge([forward, backward],mode='concat', concat_axis=1)
#modelS = Model(inputs=[x_f, x_b],output=merged)

def buildModel(sizeVectors,sizeD):
    #

    window=Sequential()
    window.add(Dense(input_dim=500, units=500, activation='softplus'))#'linear'))    

    definitionSForward=Sequential()
    definitionSForward.add(GRU(units=sizeVectors, input_shape=(sizeD,sizeVectors), return_sequences=False))
    
    definitionSBackward=Sequential() 
    definitionSBackward.add(GRU(units=sizeVectors, input_shape=(sizeD,sizeVectors), return_sequences=False))
    

    mergeD=Merge([definitionSForward,definitionSBackward],mode='concat')
    
    #merged=Merge([backwardS,extraInfoS,forwardS,mergeD],mode='concat')
    merged=Merge([window,mergeD],mode='concat')


    finalModel=Sequential()
    finalModel.add(merged)
    finalModel.add(Dense((sizeVectors*7), activation='softplus'))
    #finalModel.add(Dense(int(sizeVectors*2.5), activation='softplus'))
    finalModel.add(Dense((sizeVectors*5),activation='softplus'))
    finalModel.add(Dense((sizeVectors*2),  activation='softplus'))
    finalModel.add(Dense((sizeVectors), activation='softplus'))
    #finalModel.add(Dropout(0.2))
    finalModel.add(Dense(50, activation='softplus'))
    finalModel.add(Dense(2, activation='softplus'))
    

    if(sizeVectors==100):
        learningRate=0.0005    
    elif(sizeVectors==300):
        learningRate=0.0008
    elif(sizeVectors==500):
        learningRate=0.0010
    
    adam1=optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #finalModel.compile(loss='poisson', optimizer=adam1)#categorical_crossentropy #poisson #logcosh
    finalModel.compile(loss='mean_squared_error', optimizer=adam1)
    return finalModel


def fitModel(trainingData,finalModel,batchSize,epochsFit):
    finalModel.fit([trainingData[0],trainingData[1],trainingData[2]], trainingData[3], epochs=epochsFit, batch_size=batchSize, verbose=2)
