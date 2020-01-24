from neon.data import ArrayIterator
import numpy as np

from neon.initializers import Gaussian, Uniform
from neon.optimizers import GradientDescentMomentum
from neon.layers import Linear, Bias
from neon.layers import GeneralizedCost
from neon.transforms import SumSquared
from neon.transforms import MeanSquared
from neon.transforms import CrossEntropyMulti
from neon.transforms import Softmax, SmoothL1Loss
from neon.transforms import Logistic
from neon.transforms import Tanh
from neon.transforms import Rectlin, Identity, Explin, Normalizer
#from neon.transforms import Transform
from neon.models import Model
from neon.callbacks.callbacks import Callbacks
from neon.layers import Affine, Dropout

from neon.backends import gen_backend
#import newActivationFunction
#from newActivationFunction import MyReLu

batchSize=25
gen_backend(backend='cpu', batch_size=batchSize)

exec(open('mainTest.py').read())

X=zd
y=zd2

train = ArrayIterator(X=X, y=y, make_onehot=False)

init = Gaussian()
complementInit=Uniform()
sleep(1)
layers=[]
#

#layers.append(Affine(nout=300, init=init, activation=Tanh()))
#layers.append(Affine(nout=600, init=init, activation=Tanh()))

#layers.append(Affine(nout=300, init=init, bias=init, activation=Rectlin()))
#layers.append(Affine(nout=300, init=init, bias=init, activation=Tanh()))

layers.append(Affine(nout=100, init=init, bias=init, activation=Tanh()))
#layers.append(Affine(nout=300, init=init, bias=init, activation=Rectlin()))
#layers.append(Affine(nout=300, init=init, bias=init, activation=()))
#layers.append(Affine(nout=300, init=init, bias=init, activation=Rectlin()))

layers.append(Affine(nout=100, init=init, bias=init, activation=Tanh()))

#layers.append(Affine(nout=300, init=init, bias=init, activation=Tanh()))

#layers.append(Affine(nout=300, init=init, bias=init, activation=Identity()))

#layers.append(Affine(nout=600, init=init, activation=Tanh()))
#layers.append(Dropout(keep=0.5))

#layers.append(Affine(nout=300, init=init, bias=init, activation=Tanh()))
#layers.append(Affine(nout=300, init=init, activation=Softmax()))
cost = GeneralizedCost(costfunc=SumSquared())

mlp = Model(layers=layers)
sleep(2)
optimizer = GradientDescentMomentum(0.002, momentum_coef=0.9)
sleep(3)
mlp.fit(train, optimizer=optimizer, num_epochs=50, cost=cost, callbacks=Callbacks(mlp))

#pdict = mlp.get_description(get_weights=True)

#slope1 = mlp.get_description(True)['model']['config']['layers'][0]['params']['W']
#W1 = pdict['model']['config']['layers'][1]['params']['W']

