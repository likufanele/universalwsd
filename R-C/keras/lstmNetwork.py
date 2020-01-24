
import numpy as np
from neon.data import ArrayIterator, MNIST
from neon.initializers import Gaussian
from neon.layers import GeneralizedCost, Affine, Sequential, MergeMultistream, LSTM, MergeBroadcast,Tree
from neon.layers import BranchNode, SingleOutputTree, Dropout
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Logistic, CrossEntropyBinary
from neon.transforms import Logistic
from neon.transforms import Tanh, Softmax
from neon.transforms import Rectlin, Identity, Explin, Normalizer
from neon.layers import Linear, Bias
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser
from neon.transforms import SumSquared
from neon.backends import gen_backend

#from dataIteratorSequence import *

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

# hyperparameters
#num_epochs = args.epochs
#bnode = BranchNode()

batchSize=70
gen_backend(backend='cpu', batch_size=batchSize)

X=zd
y=zd3

train = ArrayIterator(X=[zd[:,:,0],zd2[:,:,0]], y=y, make_onehot=False, lshape=(70,100))

init_norm = Gaussian()

branch1= [LSTM(70, init_norm, activation=Tanh(), gate_activation=Logistic())]
branch2= [LSTM(70, init_norm, activation=Tanh(), gate_activation=Logistic())]
#branch1=Affine(nout=1, init=init_norm, activation=Tanh())
#branch2=Affine(nout=1, init=init_norm, activation=Tanh())

b1 = BranchNode(name="b1")
b2 = BranchNode(name="b2")

thirdBranch = [Affine(nout=1, init=init_norm, activation=Tanh()),
               Affine(nout=100, init=init_norm, activation=Tanh())]



print("before layersC")
#     Tree(layers=[branch1,branch2], alphas=[1., 1.]),
layersC = [
    #SingleOutputTree([branch1,branch2], alphas=[1.0, 1.0]),
    MergeMultistream(layers=[branch1, branch2], merge="stack"),    #branch1,
    Dropout(keep=0.5),
    Affine(nout=100, init=init_norm, activation=Tanh()),
    Affine(nout=100, init=init_norm, activation=Tanh())    
]
#layersC = SingleOutputTree(layers=[branch1,branch2],alphas=[1., 1.])
           

#Finallayers=Sequential(layersC,Linear(nout = 300, init = Gaussian()))

modelNN = Model(layers=layersC)

cost = GeneralizedCost(costfunc=SumSquared())

optimizer = GradientDescentMomentum(learning_rate=0.25, momentum_coef=0.9)

modelNN.fit(train, optimizer=optimizer, num_epochs=16, cost=cost, callbacks=Callbacks(modelNN))
