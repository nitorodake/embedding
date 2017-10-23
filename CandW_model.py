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

parser.add_argument('--epoch', '-e', default=20, type=int,
                    help='number of epochs to learn')

args = parser.parse_args()

# GPU setting
print('====================')
if args.gpu >= 0:
    chainer.cuda.get_device_from_id(args.gpu).use()
    cuda.check_cuda_available()
    print(' Use GPU  : {}'.format(args.gpu))
else:
    print(' Use CPU')

# print parameter
print('====================')
print(' Embeded Size    : {}'.format(args.embed))
print(' Hidden Unit     : {}'.format(args.unit))
print(' Window          : {}'.format(args.window))
print(' Epoch           : {}'.format(args.epoch))


#=====================
# C&W embedding model
#=====================
class CandW(chainer.Chain):
    
    def __init__(self, n_vocab, n_embed, n_units):
        super(CandW, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_embed, initialW=initializers.Uniform(1. / n_embed))
            self.l1    = L.Linear(None, n_units) # n_embed * window -> n_units
            self.l2    = L.Linear(None, 1)       # n_units -> 1
            
    def __call__(self, context, ns_context):

        # negative context
        ne = self.embed(ns_context)
        shape = ne.shape
        ne = F.reshape(ne,(shape[0],shape[1]*shape[2]))
        nh = self.l1(ne)
        nout = self.l2(F.tanh(nh))

        # positive context
        pe = self.embed(context)
        shape = pe.shape
        pe = F.reshape(pe,(1,shape[0]*shape[1]))
        ph = self.l1(pe)
        pout = self.l2(F.tanh(ph))

        shape = nout.shape
        pout = F.tile(pout,(shape[0],1))
        
        out = F.concat(( 1 - pout + nout, np.zeros((shape[0],1)).astype(np.float32)))
        return F.sum(F.max(out,axis=1))
    
    
def convert(batch):
    if args.gpu >= 0:
        batch = cuda.to_gpu(batch)
    return batch



#===============
# main process
#===============

# load learning datasets
#text_data = np.load('./data/train_text.npy')

text_data = np.random.randint(0,30000,50000).astype(np.int32)

# load vocabulary dict     
"""
with open('./data/dict/vocab.dict','rb') as fr:
    vocab = pickle.load(fr) # word2id
    n_vocab = len(vocab)
"""
n_vocab = 30000

print('====================')
print(' vocab size     : {}'.format(n_vocab))
print(' train data     : {}\n'.format(len(text_data)))

model = CandW(n_vocab, args.embed, args.unit)

if args.gpu >= 0:
    cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()

#optimizer = optimizers.Adam()
#optimizer = optimizers.AdaGrad(lr=0.1)
optimizer = optimizers.SGD()
optimizer.setup(model)


#====================
# model learning
#====================

n_data = len(text_data)
n_win = args.window

for epoch in tqdm(range(args.epoch)):
    indexes = np.arange(n_win, n_data-n_win)
    np.random.shuffle(indexes)

    for index in indexes:    
        context = []
        for offset in range(-n_win, n_win + 1):
            context.append(text_data[index + offset])        

        context = np.array(context,dtype=np.int32)
        
        ns_c_pre = np.tile(context[:n_win],(n_vocab-1,1))
        ns_c_fol = np.tile(context[-n_win:],(n_vocab-1,1))        
        ns_c_mid = np.delete(np.arange(n_vocab),text_data[index]).reshape(n_vocab-1,1)

        ns_c_tmp = np.concatenate((ns_c_pre, ns_c_mid),axis=1)
        ns_context = np.concatenate((ns_c_tmp, ns_c_fol),axis=1)


        model.zerograds()
        loss = model(convert(context),convert(ns_context.astype(np.int32)))
        loss.backward()  
        optimizer.update()


if args.gpu >= 0:
    w = cuda.to_cpu(model.embed.W.data)
else:
    w = model.embed.W.data

with open('CandW.model','wb') as fw:
    pickle.dump(w,fw)


