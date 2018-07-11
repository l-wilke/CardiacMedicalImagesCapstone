#!/usr/bin/env python

import sys
import dicom
import os
import numpy as np
import glob
import cv2 
import preproc
import argparse
from method1 import *
from method2 import *


class FindMethod(object):
    """Identify which method and type to apply normalization"""
    def __init__(self, arg):
        """

        :param arg.method: method 1 or 2
        :param arg.type: type 1, 2, or 3
        :param arg.path: source types, for identify the source dataset (check out paths in preproc)

        """
        self.arg = arg
        print (self.arg)
        self.method = self.arg.method
        self.path = self.arg.path
        self.type = self.arg.type

    def get_method(self):
        """Check preproc for valid parameters passed"""
        if self.method not in preproc.methods:
            print ("not a valid method")
            print (", ".join(preproc.methods))
            sys.exit()

        if self.path not in preproc.paths:
            print ("not a valid path")
            print (", ".join(preproc.paths))
            sys.exit()

        if self.type not in preproc.types:
            print ("not a valid type")
            print (", ".join(preproc.types))
            sys.exit()

        if self.method == '1':
            method = Method1(self.arg)
            method.main_process()
            return

        if self.method == '2':
            method = Method2(self.arg)
            method.main_process()
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', help="normalize methods:{0}".format(", ".join(preproc.methods)))
    parser.add_argument('--type', help="normalize types:{0}".format(", ".join(preproc.types)))
    parser.add_argument('--path', help="folder paths:{0}".format(", ".join(preproc.paths)))
    parser.add_argument('--source', help="sources:{0}".format(", ".join(preproc.sources.keys())))

    args = parser.parse_args()

    fm = FindMethod(args) 
    fm.get_method()
