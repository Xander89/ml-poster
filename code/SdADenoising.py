"""
 This tutorial introduces stacked denoising auto-encoders (SdA) using Theano.

 Denoising autoencoders are the building blocks for SdA.
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta"}(y) = s(W"y + b").  The weight
 matrix W" can optionally be constrained such that W" = W^T, in which case
 the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W"y + b"). The reconstruction
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML"08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""

from __future__ import print_function

import os
import sys
import timeit

import numpy
import pickle
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import copy

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from ImageDenoising import dA, loadDatasets, filterImages,saveImage
from generate_patches import get_random_patches_selection, extract_random_patches_dict

# start-snippet-1
class SdA(object):
    """Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=784,
        hidden_layers_sizes=[500, 500],
        n_outs=10,
        corruption_levels=[0.1, 0.1]
    ):
        """ This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        """

#        self.sigmoid_layers = []
        self.sigmoid_noise_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix("x", dtype="float32")  # the data is presented as rasterized images
        self.noise_x = T.matrix("noise_x", dtype="float32")
        self.y = T.ivector("y")  # the labels are presented as 1D vector of
                                 # [int] labels
        # end-snippet-1

        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP

        # start-snippet-2
        for i in range(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]
#                input_size = n_ins

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
                layer_noise_input = self.noise_x
            else:
                layer_input = self.x
                for ddAA in self.dA_layers:
                    layer_input = ddAA.get_hidden_values(layer_input)
#                theano.printing.debugprint(layer_input)
                layer_noise_input = self.sigmoid_noise_layers[-1].output
#                theano.printing.debugprint(layer_noise_input)

            sigmoid_noise_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_noise_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layers

            self.sigmoid_noise_layers.append(sigmoid_noise_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(sigmoid_noise_layer.params)
            
            # Construct a denoising autoencoder that shared weights with this
            # layer


            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          noiseInput = layer_noise_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_noise_layer.W,
                          bhid=sigmoid_noise_layer.b)
            self.dA_layers.append(dA_layer)
        # end-snippet-2
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_noise_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.cost_function(self.x)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
#        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, train_set_x_noise, batch_size):
        """ Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the dA layers
        """

        # index to a [mini]batch
        index = T.lscalar("index")  # index to a minibatch
         # % of corruption to use
        learning_rate = T.scalar("lr")  # learning rate to use
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(learning_rate)
            # compile the theano function
            #TODO remove corruption
            fn = theano.function(
                inputs=[
                    index,
                    theano.In(learning_rate, value=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end],
                    self.noise_x: train_set_x_noise[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, train_set_x, train_set_x_noise, batch_size, learning_rate):
        """Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        """

        index = T.lscalar("index")  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.noise_x: train_set_x_noise[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name="train"
        )
        return train_fn



    def get_denoised_patch_function(self, patch):
         x = patch
         for dA in self.dA_layers:
             x = dA.get_hidden_values(x)
#             z = dA.get_reconstructed_input(x)
             
         z = self.logLayer.get_denoised_patch_function(x)
         return z
#         z = self.dA_layers[-1].get_reconstructed_input(x)
#         return x

def filterImagesSdA(noise_datasets, sda):
    d = copy.deepcopy(noise_datasets)
    rgb = ("r", "g", "b")
    x = T.vector("x", dtype="float32")
    evaluate = theano.function(
        [x],
        sda.get_denoised_patch_function(x)
    )
   
    for c in rgb:
        imgs = numpy.array(d[c]["data"], dtype="float32")
        for idx in range(0, imgs.shape[0],1):
#            print("denoising: " + c + str(idx) )
            X = imgs[idx]
            Z = evaluate(X)
            d[c]["data"][idx] = Z
            
    return d

def get_cost(filtered_dataset, clean_dataset, sda):
    from logistic_sgd import get_cost_function
    rgb = ("r", "g", "b")
    x = T.vector("x", dtype="float32")
    evaluate = theano.function(
        [x],
        sda.get_denoised_patch_function(x)
    )
    
    
  
    costs = []
    da = copy.deepcopy(filtered_dataset)
    for c in rgb:    
        da[c]["data"] = numpy.zeros(da[c]["data"].shape)
        img = numpy.array(filtered_dataset[c]["data"], dtype="float32")
        img_clean = numpy.array(clean_dataset[c]["data"], dtype="float32")
        for idx in range(0, img.shape[0],1):
#            print("denoising: " + c + str(idx) )
            X = img[idx]
            X_clean = img_clean[idx]
            Z = evaluate(X)
            costs.append(get_cost_function(X_clean, Z).eval())
            da[c]["data"][idx] = Z
    print(costs)
    return da, 0#, numpy.mean(costs)
    

def unpickle(file):  
    fo = open(file, "rb")
    d = pickle.load(fo)
    fo.close()
    return d

def saveTrainedData(path, sda):
    d = {}
    d["SdA"] = {"data" : sda}
    ff = open(path, "wb")
    pickle.dump(d, ff, protocol=pickle.HIGHEST_PROTOCOL)
    ff.close()
 
def loadTrainedData(path):
    d = unpickle(path)
    
    sda = d["SdA"]["data"]
    results =(sda)
    return results   
    
#TODO change parameters to use our datasets
def test_SdA(finetune_lr=0.01, pretraining_epochs=100,
             pretrain_lr=0.01, training_epochs=100,
             hidden_layers_fraction = [0.5, 0.5, 0.5],
             noise_dataset_samples = 20, batch_size = 128
             ):

    dataset_base = "sponzat_0"
    dataset_name = dataset_base + "_10000"
    result_folder = "./result_images"
    
    
    noise_dataset_name = dataset_base +"_"+ str(noise_dataset_samples)
    clean_patches_f, noisy_patches_f, clean_datasets, noisy_datasets, patch_size = loadDatasets(dataset_name, noise_dataset_name)
    Width = patch_size[0]
    Height= patch_size[1]
    hidden_layers_sizes = [int(f*Width * Height) for f in hidden_layers_fraction]
    
    layers_string = ""
    for idx in xrange(len(hidden_layers_sizes)):
        layers_string = layers_string + "_" +str(idx)+ "L"  +str(hidden_layers_sizes[idx])
    parameters_name = ("_SdA_pretrain" + str(pretraining_epochs)+ "_tuning"+ str(training_epochs) 
                      + layers_string + "_tunerate" + str(finetune_lr) 
                      + "_pretrainrate" + str(pretrain_lr)+"_W" +str(Width)
                      +"_batchsize" + str(batch_size))
    path = "training/trained_variables_" + noise_dataset_name + parameters_name +".dat"
    train_set_x = theano.shared(clean_patches_f)
    train_set_x_noise = theano.shared(noisy_patches_f)

    isTrained =  os.path.isfile(path)
    if not isTrained:
        
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_train_batches //= batch_size
        
        # numpy random generator
        # start-snippet-3
        numpy_rng = numpy.random.RandomState(1)
        print("... building the model")
        # construct the stacked denoising autoencoder class
        sda = SdA(
            numpy_rng=numpy_rng,
            n_ins=Width * Height,
            hidden_layers_sizes=hidden_layers_sizes,
            n_outs=Width * Height
        )
        # end-snippet-3 start-snippet-4
        #########################
        # PRETRAINING THE MODEL #
        #########################
        print("... getting the pretraining functions")
         
        pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                    train_set_x_noise = train_set_x_noise,
                                                    batch_size=batch_size)
        
        print("... pre-training the model")
        start_time = timeit.default_timer()
        ## Pre-train layer-wise
        for i in range(sda.n_layers):
            # go through pretraining epochs
            for epoch in range(pretraining_epochs):
                # go through the training set
                c = []
                for batch_index in range(n_train_batches):
                    c.append(pretraining_fns[i](index=batch_index,lr=pretrain_lr))
                if epoch % 100 == 0:
                    print("Pre-training layer %i, epoch %d, cost %f" % (i, epoch, numpy.mean(c)))
                
        end_time = timeit.default_timer()
               
        print(("The pretraining code for file " +
               os.path.split(__file__)[1] +
               " ran for %.2fm" % ((end_time - start_time) / 60.)), file=sys.stderr)
        ########################
        # FINETUNING THE MODEL #
        ########################
        
        # get the training, validation and testing function for the model
        print("... getting the finetuning functions")
        train_fn = sda.build_finetune_functions(
            train_set_x = train_set_x,
            train_set_x_noise = train_set_x_noise,
            batch_size=batch_size,
            learning_rate=finetune_lr
        )
        
        print("... finetunning the model")
        
        start_time = timeit.default_timer()
        
        
        epoch = 0
        
        while (epoch < training_epochs): # and (not done_looping)
            epoch = epoch + 1
            c = []
            for minibatch_index in range(n_train_batches):
                c.append(train_fn(minibatch_index))
            if epoch % 100 == 0:
                print("fine tuning, epoch %d, cost %f" % (epoch, numpy.mean(c)))
        
        end_time = timeit.default_timer()
        
        print(("The training code for file " +
               os.path.split(__file__)[1] +
               " ran for %.2fm" % ((end_time - start_time) / 60.)), file=sys.stderr)
    if isTrained:
        sda = loadTrainedData(path)

    d = filterImagesSdA(noisy_datasets, sda)

    
    saveTrainedData(path, sda)
    saveImage(d, noise_dataset_name  + parameters_name,
                                     result_folder)
#    d2, cost = get_cost(noisy_datasets, clean_datasets, sda)
#    # end-snippet-4
if __name__ == "__main__":
    pretrain_epochs = [10000]
    finetune_rates = [0.01]
    pretrain_rates = [0.1]
    finetune_epochs = [10000]    
    hl = [[0.5, 0.5, 0.5]]
    batch_sizes = [128]
    noise_data_samples = [20]
    for n in noise_data_samples:
        for b in batch_sizes:
            for l in hl:
                for lr2 in finetune_rates:
                    for lr1 in pretrain_rates:
                        for e1 in pretrain_epochs:
                            for e2 in finetune_epochs:                       
                                test_SdA(finetune_lr=lr2, pretraining_epochs=e1, pretrain_lr=lr1, training_epochs=e2, hidden_layers_fraction = l, noise_dataset_samples = n,  batch_size = b)
