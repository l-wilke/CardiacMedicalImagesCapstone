#!/usr/bin/env python


""" Ensamble process, read in a config file, i.e. post_process_config and to combine prediction results accordingly before the contour removal process """

import glob
from helper_utilities import *


def do_post_processing(args):
    """
    Combine predictions to either apply average or majority vote to improve result

    Args:
      args: args passed in from the config file

    Returns: none

    """
    arg_list = ['input_dir','base_dir','output_dir','systolic_path','diastolic_path','sources','volume_dir','diastolic_models','systolic_models','ensamble_type']

    dir_args = dir(args)

    for x in arg_list:
        if x not in dir_args:
            print ("insufficient arguments ")
            print ("enter {0} in the config file".format(",".join(arg_list)))
            return

    do_ensamble(args)


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print ('Provide a config file')

    myconfig = sys.argv[1]

    if myconfig.endswith('.py'):
        myconfig = myconfig.replace('.py','')

    args = __import__(myconfig)

    do_post_processing(args)
