import numpy as np
from utils import gen_batches, params_unpack, params_pack
from sigmoid import sigmoid, sigmoid_grad
from softmax import softmax

class NeuralNetwork(object):

    def __init__(self, 
                hidden_layer_sizes=(3, ), 
                learning_rate=0.1, 
                batch_size=10, 
                epoch=1000, 
                momentum=0,
                tol=1e-10,
                reg_coef=0.0001,
                verbose=False):
        """The 

        Parameters:

        hidden_layer_sizes -- the number of hidden layer units exclude input and output layer
        learning_rate -- we use gradient decent algorightm to do optimization
        batch_size -- 
        epoch --
        momentum -- 
        tol -- the exit tolerrence of loss changes about two continuous epoch
        reg_coef -- the coefficient of regularation 
        verbose -- show Iteration and loss infomation
        """
        

        self.learning_rate = learning_rate
        self.max_iter = epoch
        self.momentum = momentum
        self.batch_size = batch_size
        self.tol = tol
        self.verbose = verbose
        self.hidden_layer_sizes = hidden_layer_sizes
        self.reg_coef = reg_coef


        # self.W_, self.b_ = [], []
        self.loss_ = []

        self._activation_func = sigmoid
        self._out_loss_func = {
            'out':softmax, 
            'loss': lambda a,y: (-y*np.log(a)).sum()
        }

    def fit(self, X, y):
        """Training NeuralNetwork to fit the function of  X -> y
        
        Parameters:
        X -- samples
        y -- labels

        Return:
        self
        """
        return self._fit(X, y)

    def predict(self, X):
        """Do prediction

        Parameters:
        X -- samples
        
        Return:
        labels
        """
        if not hasattr(self, 'W_'):
            raise Exception('You should call `fit` before `predict`')
        
        acivations = self._forward_pass(X, self.W_, self.b_)
        n_layers = len(acivations)

        y_pred = acivations[n_layers - 1].argmax(axis=1)
        n_label = np.max(y_pred) + 1
        y_pred = np.eye(n_label,dtype=np.int64)[y_pred]
        return y_pred

    def _initilize(self, layer_sizes):
        # initial W and b
        self.W_, self.b_ = [], []
        for i in range(len(layer_sizes) - 1):
            W, b = self._init_W_b(layer_sizes[i], layer_sizes[i + 1])
            self.W_.append(W)
            self.b_.append(b)

        self.loss_ = []

    def _init_W_b(self, n_in, n_out):
        bound = 2.0 / (n_in + n_out)
        W = np.random.uniform(-bound, bound, (n_in, n_out))
        b = np.random.uniform(-bound, bound, (1, n_out))
        return W, b


    def _forward_pass(self, X, W, b):
        """Forward propgation

        Parameters:
        X -- samples
        W -- weights
        b -- bias

        Return:
        activations values of every units in the net
        """
        assert len(W) == len(b)
        n_layers = len(W) + 1

        activations = [0]*n_layers
        activations[0] = X
        for l in range(0, n_layers - 1):
            # layer l -> layer l+1
            z = np.dot(activations[l], W[l]) + b[l]
            if l + 1 < n_layers - 1:
                # l+1 is hidden layer
                activations[l + 1] = self._activation_func(z)
            else:
                # l+1 is output layer
                activations[l + 1] = self._out_loss_func['out'](z)

        return activations

    def _backprop(self, X, y, W, b, activations):
        """Back propgation

        Parameters:
        X -- samples
        y -- labels
        W -- weights
        b -- bias
        acivations - activation values of every units in the net

        Return:
        loss, gradients of W, gradients of b
        """
        deltas = [0]*len(activations)

        m = X.shape[0]
        
        n_layers = len(activations)

        grad_W, grad_b = [0]*len(W), [0]*len(b)

        regularation = 0

        loss = self._out_loss_func['loss'](activations[n_layers - 1], y)/m

        deltas[n_layers - 1] = (activations[n_layers - 1] - y)/m

        for i in range(n_layers - 2, -1, -1):
            grad_W[i] = np.dot(activations[i].T, deltas[i + 1])
            grad_W[i] += W[i]*self.reg_coef/m
            grad_b[i] = deltas[i + 1].sum(axis=0)
            if i > 0:
                deltas[i] = np.dot(deltas[i + 1], W[i].T)
                deltas[i] *= sigmoid_grad(activations[i])
            regularation += np.power(W[i], 2).sum()
        
        regularation *= self.reg_coef/(2*m)

        loss += regularation

        return loss, grad_W, grad_b

    def _validate_input(self, X, y):
        

        return X, y

    def _validate_hyperparams(self,X, y):
        if not (0 < self.batch_size < X.shape[0]):
            raise Exception('Batch size must integer and gt 0 and lt the number of samples, but {} got'.format(str(self.batch_size)))

        if len(self.hidden_layer_sizes) == 0:
            raise Exception('There must have one hidden layer at least, but {} got.'.format(self.hidden_layer_sizes))
        
        if not (0 <= self.momentum <= 1):
            raise Exception('The momentum should be gte 0 and lte 1, but {} got.'.format(self.hidden_layer_sizes))
    
    def _pack_W_b(self, W, b):
        params = []
        for i in range(len(W)):
            params += [W[i].flatten(), b[i].flatten()]
        return np.concatenate(params)

    def _unpack_W_b(self, params, W, b):
        ans_W, ans_b = [], []
        ofs = 0
        for i in range(len(W)):
            length =W[i].shape[0] * W[i].shape[1]
            ans_W.append(np.reshape(params[ofs: ofs+length], (W[i].shape[0], W[i].shape[1])))
            ofs += length

            length = b[i].shape[0] * b[i].shape[1]
            ans_b.append(np.reshape(params[ofs: ofs+length], (b[i].shape[0], b[i].shape[1])))
            ofs += length
        return ans_W, ans_b


    def _fit(self, X, y):
        hidden_layer_sizes = self.hidden_layer_sizes

        if not hasattr(hidden_layer_sizes, '__iter__'):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        self._validate_hyperparams(X, y)

        X, y = self._validate_input(X, y)

        # make y 2D
        if len(y.shape) == 1:
            y.reshape(y.shape[0], 1)

        # tell the size of every layers
        layer_sizes = [X.shape[1]]
        layer_sizes.extend(hidden_layer_sizes)
        layer_sizes.append(y.shape[1])

        self._initilize(layer_sizes)

        packed_params = self._pack_W_b(self.W_, self.b_)

        # trainning params assignment
        batch_size = self.batch_size
        learning_rate = self.learning_rate
        momentum = self.momentum
        tol = self.tol
        verbose = self.verbose
        velocities = np.zeros_like(packed_params)
        self.loss_ = []

        for i in range(self.max_iter):
            epoch_loss = 0

            for batch_slice in gen_batches(X.shape[0], batch_size):
                _X, _y = X[batch_slice], y[batch_slice]

                activations = self._forward_pass(_X, self.W_, self.b_)

                loss, grad_W, grad_b = self._backprop(_X, _y, self.W_, self.b_, activations)

                packed_grad = self._pack_W_b(grad_W, grad_b)
                packed_params = self._pack_W_b(self.W_, self.b_)

                # apply gradients
                updates = momentum*velocities - learning_rate*packed_grad
                velocities = updates
                packed_params = packed_params + updates

                self.W_, self.b_ = self._unpack_W_b(packed_params, self.W_, self.b_)

                epoch_loss += loss * (batch_slice.stop - batch_slice.start)
            
            epoch_loss /= X.shape[0]
            self.loss_.append(epoch_loss)

            # two continuous epoch get loss changes lower than tol
            if i > 2 and \
                (self.loss_[i] - self.loss_[i-1] > -tol) and \
                (self.loss_[i-1] - self.loss_[i-2] > -tol):
                break

            if verbose:
                print('Iteration {:d}, loss = {:f}'.format(i+1, epoch_loss))

        return self
