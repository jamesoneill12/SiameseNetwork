from collections import OrderedDict
from rnn import RNN, MetaRNN
import numpy as np
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano as T
import theano.tensor as tensor

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)

class SiameseNetwork(RNN,RNN):

    def __init__(self,num_class,n_hidden = 10,n_in = 5,n_out = 3,
                 n_steps = 10,n_seq = 1000,loss_function='dtw',attention=False):

        super(SiameseNetwork,self).\
            __init__(n_hidden = n_hidden,n_in = n_in,n_out = n_out,
                 n_steps = n_steps,n_seq = n_seq)

        self.loss_function = loss_function
        self.num_class = num_class
        self.attention = attention

    def init_params(self,options):
            """
            Global (not LSTM) parameter. For the embedding and the classifier.
            """
            params = OrderedDict()
            # embedding
            randn = np.random.rand(options['n_words'],
                                      options['dim_proj'])
            params['Wemb'] = (0.01 * randn).astype(config.floatX)
            params = get_layer(options['encoder'])[0](options,
                                                      params,
                                                      prefix=options['encoder'])
            # classifier
            params['U'] = 0.01 * np.random.randn(options['dim_proj'],
                                                    options['ydim']).astype(config.floatX)
            params['b'] = np.zeros((options['ydim'],)).astype(config.floatX)

            return params

    def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
        nsteps = state_below.shape[0]
        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1

        assert mask is not None

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        def _step(m_, x_, h_, c_):
            preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
            preact += x_

            i = T.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
            f = T.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
            o = T.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
            c = T.tanh(_slice(preact, 3, options['dim_proj']))

            c = f * c_ + i * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_

            h = o * tensor.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_

            return h, c

    def feedforward(self):
        return 0

    def backprop(self,type='sgd'):
        return 0

    def grad_desc(self,options):
        return 0
