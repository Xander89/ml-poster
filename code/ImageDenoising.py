# -*- coding: utf-8 -*-
"""
Created on Fri Jul 01 15:52:10 2016

@author: aluo
"""

from __future__ import print_function


import os
import sys
import errno
import timeit
import pickle
import numpy
from matplotlib import pyplot
from generate_patches import recombine_image
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import load_data
from utils import tile_raster_images
try:
    import PIL.Image as Image
except ImportError:
    import Image

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
							

class dA(object):
    
    
    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        noiseInput=None,
        n_visible=32*32,
        n_hidden=800,
        W=None,
        bhid=None,
        bvis=None
    ):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None


        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input
        if noiseInput is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.noise_x = T.dmatrix(name='noiseInput')
        else:
            self.noise_x = noiseInput
            
        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_denoised_patch_function(self, patch):
         y = self.get_hidden_values(patch)
         z = self.get_reconstructed_input(y)
         return z
        
    def get_cost_updates(self, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

#        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        
        tilde_x=self.noise_x
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L)
#        cost = L 
#        square_param = numpy.multiply(self.params[0],self.params[0])
#        regularization = learning_rate* 0.5 * T.mean(T.sum(T.sum(square_param,axis = 0),axis=0))
        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
 
        print(gparams)
        #gparams[0] = gparams[0] + learning_rate * self.params[0] / self.params[0].size
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)

def test_dA(Width = 32, Height = 32, hidden = 800, learning_rate=0.1, training_epochs=15,
            dataset = None, noise_dataset=None,
            batch_size=20, output_folder='dA_plots'):

    """
    This demo is tested on MNIST

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type dataset: string
    :param dataset: path to the picked dataset

    """

    train_set_x = theano.shared(dataset)
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
   
    # start-snippet-2
    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x', dtype='float32')  # the data is presented as rasterized images
    noise_x = T.matrix('noise_x', dtype='float32')
    # end-snippet-2

#    if not os.path.isdir(output_folder):
#        os.makedirs(output_folder)
#    os.chdir(output_folder)

 
    #####################################
    # BUILDING THE MODEL CORRUPTION 30% #
    #####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    noise_train_set_x = theano.shared(noise_dataset)
    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        noiseInput=noise_x,
        n_visible=Width * Height,
        n_hidden=hidden
    )

    cost, updates = da.get_cost_updates(
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            noise_x: noise_train_set_x[index * batch_size: (index + 1) * batch_size] 
        }
    )

    start_time = timeit.default_timer()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in range(training_epochs):
        # go through trainng set
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))

        if epoch % 100 == 0:
            print('Training epoch %d, cost ' % epoch, numpy.mean(c))

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

#    print(('The 30% corruption code for file ' +
#           os.path.split(__file__)[1] +
#           ' ran for %.2fm' % (training_time / 60.)), file=sys.stderr)
    # end-snippet-3
#
#    # start-snippet-4
#    image = Image.fromarray(tile_raster_images(
#        X=da.W.get_value(borrow=True).T,
#        img_shape=(Width, Height), tile_shape=(10, 10),
#        tile_spacing=(1, 1)))
#    image.save('filters_corruption_30.png')
#    # end-snippet-4
#
#    os.chdir('../')
    W_corruption = da.W
    bhid_corruption = da.b
    bvis_corruption = da.b_prime
    results = (W_corruption,
               bhid_corruption, bvis_corruption)
    return results


def unpickle(file):
    
    fo = open(file, 'rb')
    d = pickle.load(fo)
    fo.close()
    return d
    
def showRGBImage(array_data, W, H):
    array_data = array_data.reshape(3,W, H).transpose()
    array_data = numpy.swapaxes(array_data,0,1)
    pyplot.axis('off')
    array_data = pyplot.imshow(array_data)


def showGrayImage(data, W, H):
    data = data.reshape(W,H)
    pyplot.axis('off')
    pyplot.imshow(data,cmap='Greys_r')
    

def showEncodeImage(data, autoEncoder, W, H):
    X = data
    tilde_X = X
    Y = autoEncoder.get_hidden_values(tilde_X)
    Z = autoEncoder.get_reconstructed_input(Y)
    Y = Y.eval()
    Z = Z.eval()
#    tilde_X = tilde_X.eval()
    showGrayImage(tilde_X, W, H)
    pyplot.figure()
    showGrayImage(Z, W, H)
    pyplot.figure()
    pyplot.show()
    
def saveTrainedData(path,noise_W, noise_b, noise_b_p,hidden, Width, Height ):
    d = {}
    d["noise_W"] = {"data" : noise_W}
    d["noise_b"] = {"data" : noise_b}
    d["noise_b_p"] = {"data" : noise_b_p}
    d["hidden"] = {"data" : hidden}
    d["Width"] = {"data" : Width}
    d["Height"] = {"data" : Height}
    ff = open(path, "wb")
    pickle.dump(d, ff)
    ff.close()
    
def loadTrainedData(path):
    d = unpickle(path)
    noise_W = d["noise_W"]["data"]
    noise_b = d["noise_b"]["data"]
    noise_b_p = d["noise_b_p"]["data"]
    hidden = d["hidden"]["data"]
    Width = d["Width"]["data"]
    Height = d["Height"]["data"]
    results =(noise_W,noise_b,noise_b_p,hidden,Width,Height)
    return results
    
def filterImages(noise_datasets, autoEncoder):
    d = noise_datasets.copy()
    rgb = ('r', 'g', 'b')
    x = T.vector('x', dtype='float32')
    evaluate = theano.function(
        [x],
        autoEncoder.get_denoised_patch_function(x)
    )
    print(autoEncoder.W.eval().shape, autoEncoder.W_prime.eval().shape)
    for c in rgb:
        imgs = numpy.array(d[c]['data'], dtype='float32')
        for idx in range(0, imgs.shape[0],1):
#            print("denoising: " + c + str(idx) )
            X = imgs[idx]
            Z = evaluate(X)
            d[c]['data'][idx] = Z
            
    return d

def saveImage(image_dict, image_file_name, results_folder="./result_images"):
    recombine_image(image_dict, results_folder + os.sep +image_file_name + '.png')
    
    

def loadDatasets(reference_name, noisy_dataset_name,source_folder = "./image_patch_data",results_folder="./result_images"):
    make_sure_path_exists(source_folder)
    make_sure_path_exists(results_folder)
    
    clean_dataset_path = source_folder + os.sep + reference_name + '.dat'
    clean_datasets = unpickle(clean_dataset_path)
    clean_patches = numpy.concatenate((clean_datasets['r']['data'], clean_datasets['g']['data'], clean_datasets['b']['data']),axis=0)
    clean_patches_f = numpy.array(clean_patches, dtype='float32')

    noisy_dataset_path = source_folder + os.sep + noisy_dataset_name +'.dat'
    noisy_datasets = unpickle(noisy_dataset_path)
    noisy_patches = numpy.concatenate((noisy_datasets['r']['data'],noisy_datasets['g']['data'],noisy_datasets['b']['data']),axis=0)
    noisy_patches_f = numpy.array(noisy_patches, dtype='float32')
    
    return clean_patches_f, noisy_patches_f, clean_datasets, noisy_datasets

if __name__ == '__main__':
   
    dataset_base = "rendering"
    dataset_name = dataset_base + "_10000"
    result_folder = "./result_images"
    
    noise_dataset_samples = 5
    noise_dataset_name = dataset_base +'_'+ str(noise_dataset_samples)
    clean_patches_f, noisy_patches_f, clean_datasets, noisy_datasets = loadDatasets(dataset_name, noise_dataset_name)
    
    Width = Height = 32
    hidden = Width * Height * 2 // 3

    training_epochs = 20
    learning_rate =0.01
    batch_size = clean_patches_f.shape[0]

    path = 'training/trained_variables_' + dataset_base + '_'+ str(noise_dataset_samples) +'_' + str(training_epochs)+'.dat'
    isTrained =  os.path.isfile(path)

    if not isTrained:
        noise_W, noise_b, noise_b_p = test_dA(dataset=clean_patches_f,learning_rate=learning_rate,
                                                training_epochs=training_epochs,hidden=hidden,
                                                Width = Width, Height = Height,
                                                batch_size = batch_size,
                                                noise_dataset=noisy_patches_f)
        saveTrainedData(path, noise_W, noise_b, noise_b_p,hidden, Width, Height )
    else:
        noise_W, noise_b, noise_b_p,hidden, Width, Height = loadTrainedData(path)
    

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    noiseDA = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=clean_patches_f,
        noiseInput=noisy_patches_f,
        n_visible=Width * Height,
        n_hidden=hidden,
        W=noise_W,
        bhid=noise_b,
        bvis=noise_b_p
    )
    denoised_datasets = filterImages(noisy_datasets,noiseDA)
    saveImage(denoised_datasets, noise_dataset_name + "_" + str(training_epochs),
                                     result_folder)

