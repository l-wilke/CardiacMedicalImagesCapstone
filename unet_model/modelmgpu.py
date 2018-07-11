#!/usr/bin/env python

"""This class allows us to run our training and predicting processes with multi-gpus
   This code is borrowed from https://github.com/keras-team/keras/issues/2436#issuecomment-354882296
"""

from keras.models import *
from keras.utils import multi_gpu_model

class ModelMGPU(Model):
    """ The based model is passed in for running the process with multi gpus when possible """
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)

