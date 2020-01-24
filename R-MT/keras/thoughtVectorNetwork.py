from neon.data import ArrayIterator
import numpy as np

from neon.initializers import Gaussian,Uniform
from neon.optimizers import GradientDescentMomentum
from neon.layers import Linear, Bias
from neon.layers import GeneralizedCost
from neon.transforms import SumSquared, MeanSquared
from neon.transforms import CrossEntropyMulti
from neon.transforms import Softmax,SmoothL1Loss
from neon.transforms import Logistic
from neon.transforms import Identity
from neon.transforms import Tanh
from neon.transforms import Rectlin, Identity

from neon.models import Model
from neon.callbacks.callbacks import Callbacks
from neon.layers import Affine,Dropout

from neon import logger as neon_logger

from neon.initializers import Uniform, GlorotUniform
from neon.layers import (GeneralizedCost, LSTM, Affine, Dropout, LookupTable,
                         RecurrentSum, Recurrent, DeepBiLSTM, DeepBiRNN)

from neon.backends import gen_backend

from neon.transforms import Logistic, Tanh, Softmax, CrossEntropyMulti, Accuracy

from neon.util.argparser import NeonArgparser, extract_valid_args

parser = NeonArgparser(__doc__)
parser.add_argument('--rlayer_type', default='lstm',
                    choices=['bilstm', 'lstm', 'birnn', 'bibnrnn', 'rnn'],
                    help='type of recurrent layer to use (lstm, bilstm, rnn, birnn, bibnrnn)')

args = parser.parse_args(gen_be=False)

hidden_size = 75
embedding_dim = 300
reset_cells = True
#vocab_size = 12000
sentence_length = 251

uni = Uniform(low=-0.1 / embedding_dim, high=0.1 / embedding_dim)
init_norm = Gaussian()
g_uni = GlorotUniform()

X=zd
y=zd2

train = ArrayIterator(X=zd, y=zd2, make_onehot=False)

rlayer = LSTM(hidden_size, init_norm, activation=Tanh(),
                  gate_activation=Logistic(), reset_cells=reset_cells)

llayer = LSTM(hidden_size, init_norm, activation=Tanh(),
                  gate_activation=Logistic(), reset_cells=reset_cells)


layersC = [MergeMultistream(layers=[llayer, rlayer], merge="stack"),
          Affine(nout=300, init=init_norm, activation=Tanh())]

layers=Sequential(layersC,Linear(nout = 300, init = Gaussian()))


modelR = Model(layers=layers)
cost = GeneralizedCost(costfunc=SumSquared())
optimizer = GradientDescentMomentum(0.02, momentum_coef=0.9)

callbacks = Callbacks(model, **args.callback_args)

# train model
modelR.fit(train, optimizer=optimizer, num_epochs=10, cost=cost, callbacks=Callbacks(modelR))

#pdict = mlp.get_description(get_weights=True)
