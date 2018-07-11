
The preprocessing directory is mainly where the normalization scripts are

There are different ways to apply normalization

There are 2 methods and each method consists of 4 types of normalization

Method 1, type 0: Orientation, rescaling, cropping, and applying contrast 

Here's an example of how to normalize acdc data for method 1 type 0

python3 find_method.py --method 1 --type 0 --path train --source acdc

For valid methods, paths, sources, please check out preproc.py

We assume the files reside in masvol, rook volume

An example of the input path would be /masvol/data/acdc/niftidata



