"""
C&W word embedding model.

Collobert, Ronan, and Jason Weston. 
"A unified architecture for natural language processing: Deep neural networks with multitask learning." 
Proceedings of the 25th international conference on Machine learning. ACM, 2008.

by chainer v 2.x
"""

import argparse
import collections
import numpy as np
import pickle
from tqdm import tqdm

import chainer
from chainer import cuda, optimizers, initializers
import chainer.links as L
import chainer.functions as F
from chainer.utils import walker_alias

# parser setting
parser = argparse.ArgumentParser()

parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID(negative value indicates CPU)')

parser.add_argument('--embed', '-em', default=50, type=int,
                    help='number of embeded size')

parser.add_argument('--unit', '-u', default=20, type=int,
                    help='number of hidden units')

parser.add_argument('--window', '-w', default=3, type=int,
                    help='window size')

parser.add_argument('--batchsize', '-b', type=int, default=1000,
                    help='learning minibatch size')

parser.add_argument('--negative-size', '-ns', default=100, type=int,
                    help='number of negative samples')

parser.add_argument('--epoch', '-e', default=5, type=int,
                    help='number of epochs to learn')

args = parser.parse_args()

# GPU setting
print('====================')
if args.gpu >= 0:
    chainer.cuda.get_device_from_id(args.gpu).use()
    cuda.check_cuda_available()
    xp = cuda.cupy
    print(' Use GPU  : {}'.format(args.gpu))
else:
    xp = np
    print(' Use CPU')

# print parameter
print('====================')
print(' Embeded Size      : {}'.format(args.embed))
print(' Hidden Unit       : {}'.format(args.unit))
print(' Window            : {}'.format(args.window))
print(' Epoch             : {}'.format(args.epoch))
print(' Minibatch size    : {}'.format(args.batchsize))
print(' Sampling Size     : {}'.format(args.negative_size))


#=====================
# C&W enbedding model
#=====================
class CandW(chainer.Chain):
    
    def __init__(self, n_vocab, n_embed, n_units):
        super(CandW, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_embed, initialW=initializers.Uniform(1. / n_embed))
            self.l1    = L.Linear(n_embed*(args.window*2+1), n_units) # n_embed * (window*2 + 1)-> n_units
            self.l2    = L.Linear(n_units, 1)       # n_units -> 1
            
    def __call__(self, context, context_pre, context_fol, neg_context):
        bs = context.shape[1]
        
        # ----- context ---------------- 
        # ---------- positive ----------
        e = self.embed(context)
        e = F.concat(e,axis=1)
        h = self.l1(e)
        th = F.tanh(h)
        pout = self.l2(th)
        pout = F.tile(pout,(args.negative_size,1))

        # ---------- negative ----------
        # embedding
        pe = self.embed.W.data[context_pre.T]
        fe = self.embed.W.data[context_fol.T]

        shape = pe.shape
        pe = pe.reshape(shape[0],shape[1]*shape[2])
        fe = fe.reshape(shape[0],shape[1]*shape[2])
    
        pe = xp.tile(pe,(args.negative_size,1))
        fe = xp.tile(fe,(args.negative_size,1))
        
        #ne = xp.array([self.embed.W.data[val] for val in neg_context])
        ne = xp.array(self.embed.W.data[neg_context])

        # concatenate        
        tmp = xp.concatenate((pe,ne),axis=1)
        ne_in = xp.concatenate((tmp,fe),axis=1) 

        # forward 
        nout = xp.tanh(ne_in.dot(self.l1.W.data.T))
        nout = nout.dot(self.l2.W.data.T)
                
        # ---------- hinge loss calculate ----------
        loss = F.hinge(pout + nout, xp.zeros((args.negative_size*bs,),dtype=np.int32))
        print(loss * args.negative_size * bs)
        #return loss
        return loss * args.negative_size * bs
        

#===============
# main process
#===============

# load learning datasets
text_data = np.load('./data/sentence_data.npy')
counts = collections.Counter(text_data)
cs = [counts[w] for w in range(len(counts))]

# load vocabulary dict     
with open('./data/vocab.dict','rb') as fr:
    vocab = pickle.load(fr) # word2id
    n_vocab = len(vocab)
    
print('====================')
print(' vocab size     : {}'.format(n_vocab))
print(' train data     : {}\n'.format(len(text_data)))

model = CandW(n_vocab, args.embed, args.unit)

if args.gpu >= 0:
    cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()

#optimizer = optimizers.Adam()
optimizer = optimizers.AdaGrad()
#optimizer = optimizers.SGD()
optimizer.setup(model)
#optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

#====================
# model learning
#====================

sampler = walker_alias.WalkerAlias(np.power(cs,0.75))
ng_size = args.negative_size
n_data = len(text_data)
n_win = args.window
bs = args.batchsize

for epoch in tqdm(range(args.epoch)):
    indexes = np.arange(n_win, n_data-n_win)
    np.random.shuffle(indexes)

    for n in range(0, len(indexes), bs):
        index = indexes[n:n+bs]

        context = []

        for offset in range(-n_win, n_win + 1):                
            context.append(text_data[index + offset])
            
                
        context = np.array(context,dtype=np.int32)
        context_pre = context[:n_win]
        context_fol = context[-n_win:]
        neg_context = np.array(sampler.sample(ng_size * len(index)),dtype=np.int32)
        
        
        # convert
        if args.gpu >= 0:
            context = cuda.to_gpu(context)
            context_pre = cuda.to_gpu(context_pre)
            context_fol = cuda.to_gpu(context_fol)
            neg_context = cuda.to_gpu(neg_context)
            
        model.zerograds()
        loss = model(context, context_pre, context_fol, neg_context)
        loss.backward()  
        optimizer.update()


if args.gpu >= 0:
    w = cuda.to_cpu(model.embed.W.data)
else:
    w = model.embed.W.data

with open('SSWE.model','wb') as fw:
    pickle.dump(w,fw)
