#!/usr/bin/env python

"""Configuration variables for preprocessing"""

methods = ['1','2']
types = ['0','1','2','3']
paths = ['train','validate','test','challenge_online','challenge_validation','challenge_training','niftidata','challenge']
dsbpaths = ['train','validate','test']
sunnybrookpaths = ['challenge_online', 'challenge_validation','challenge_training','challenge']
acdcpaths = ['niftidata']

re_patterns = {"acdc":"([^/]*)/patient\d+_slice(\d+)_frame(\d+)_label_fix.nii.npy",
               "sunnybrook":"([^/]*)/IM-(\d{4})-(\d{4}).dcm.label.npy",
               "dsb":"([^/]*)/sax-(\d+)_(\d{4}-?\d{4}?).dcm.npy"}

labels = {"acdc":'label_fix.nii.npy',
          "sunnybrook":'IM-*dcm.label.npy',
          "dsb":''}

filenames = {"acdc":"%s_slice%s_frame%d.nii.npy",
             "sunnybrook":"IM-%s-%04d.dcm.npy",
             "dsb":"sax_%s-%s.dcm.npy"}

sources = {"dsb":{"dir":"/masvol/data/dsb",
                  "paths":dsbpaths,
                  "string":"study",
                  "pattern":"sax",
                 },
           "sunnybrook":{"dir":"/masvol/data/sunnybrook",
                         "paths":sunnybrookpaths,
                         "string":"*",
                         "pattern":"",
                        },
           "acdc":{"dir":"/masvol/data/acdc",
                   "paths":acdcpaths,
                   "string":"",
                   "pattern":"",
                  },
          }

normoutputs = {"dsb":{"dir":"/masvol/output/dsb/norm",
                 },
           "sunnybrook":{"dir":"/masvol/output/sunnybrook/norm",
                        },
           "acdc":{"dir":"/masvol/output/acdc/norm",
                  },
          }

sax_series_all = {
    'SC-HF-I-1': '0004',
    'SC-HF-I-2': '0106',
    'SC-HF-I-4': '0116',
    'SC-HF-I-5': '0156',
    'SC-HF-I-6': '0180',
    'SC-HF-I-7': '0209',
    'SC-HF-I-8': '0226',
    'SC-HF-I-9': '0241',
    'SC-HF-I-10': '0024',
    'SC-HF-I-11': '0043',
    'SC-HF-I-12': '0062',
    'SC-HF-I-40': '0134',
    'SC-HF-NI-3': '0379',
    'SC-HF-NI-4': '0501',
    'SC-HF-NI-7': '0523',
    'SC-HF-NI-12': '0286',
    'SC-HF-NI-11': '0270',
    'SC-HF-NI-13': '0304',
    'SC-HF-NI-14': '0331',
    'SC-HF-NI-15': '0359',
    'SC-HF-NI-31': '0401',
    'SC-HF-NI-33':'0424',
    'SC-HF-NI-34': '0446',
    'SC-HF-NI-36': '0474',
    'SC-HYP-1': '0550',
    'SC-HYP-3': '0650',
    'SC-HYP-6': '0767',
    'SC-HYP-7': '0007',
    'SC-HYP-8': '0796',
    'SC-HYP-9': '0003',
    'SC-HYP-10': '0579',
    'SC-HYP-11': '0601',
    'SC-HYP-12': '0629',
    'SC-HYP-37': '0702',
    'SC-HYP-38': '0734',
    'SC-HYP-40': '0755',
    'SC-N-2': '0898',
    'SC-N-3': '0915',
    'SC-N-5': '0963',
    'SC-N-6': '0981',
    'SC-N-7': '1009',
    'SC-N-9': '1031',
    'SC-N-10': '0851',
    'SC-N-11': '0878',
    'SC-N-40': '0944',
}
