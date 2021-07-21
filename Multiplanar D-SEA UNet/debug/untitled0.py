# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 17:57:37 2020

@author: kyon
"""

import numpy as np
import nibabel as nib
import glob

def _audit_classes(nii_lab_paths, logger=None):
    print("Auditing number of target classes. This may take "
           "a while as data must be read from disk."
           "\n-- Note: avoid this by manually setting the "
           "n_classes attribute in train_hparams.yaml.")
    # Select up to 50 random images and find the unique classes
    lab_paths = np.random.choice(nii_lab_paths,
                                 min(50, len(nii_lab_paths)),
                                 replace=False)
    classes = []
    for l in lab_paths:
        classes.extend(np.unique(nib.load(l).get_data()))
    classes = np.unique(classes)
    n_classes = classes.shape[0]

    # Make sure the classes start from 0 and step continuously by 1
    c_min, c_max = np.min(classes), np.max(classes)
    if c_min != 0:
        raise ValueError("Invalid class audit - Class integers should"
                         " start from 0, found %i (classes found: %s)"
                         % (c_min, classes))
    #if n_classes != max(classes) + 1:
    #    raise ValueError("Invalid class audit - Found %i classes, but"
     #                    " expected %i, as the largest class value"
     ##                    " found was %i. Classes found: %s"
      #                   % (n_classes, c_max+1, c_max, classes))
    return n_classes
labels = []
for i in glob.glob('labels/*'):
    labels.append(i)
print(_audit_classes(labels))

#a = np.unique(nib.load('labels/1.nii.gz').get_data())
#b = np.unique(nib.load('seg.nii.gz').get_data())