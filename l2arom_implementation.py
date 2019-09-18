import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.feature_selection import RFE

import warnings
warnings.filterwarnings("ignore")

##
# X is numpy array witht the data (rows are data instances)
# Y is a numpy vector with the class labels (-1 or 1)
# C is the regularization coefficient of the SVM
# threshold is the threshold value to drop features in L2AROM

def variable_ranking(X, Y, C = 1, threshold = 1e-10):
    """
    """

    # Copy X to modify it later

    final_X = X.copy()

    z = np.ones(X.shape[1])

    # Number of attributes

    length = z.shape[0]

    # Array that stores the elimination order, being the higher number the first attribute 
    # that is eliminated and 1 the last one

    elimination_order = np.zeros(length, dtype = int)
    original_feature_indices = np.arange(0, length, dtype = int)
    clf = SVC(kernel = "linear", C = C, random_state = 0)

    iter_without_dropping = 0
    n_removed_features = 0

    while iter_without_dropping < 20 and length > 10:

        # Fit the SVC and compute z

        clf.fit(final_X * np.outer(np.ones(X.shape[ 0 ]), z), Y)
        z *= np.abs(clf.coef_[0]) # In absolute value

        n_features_to_drop = np.sum(z < threshold)
        
        if n_features_to_drop == 0:
            iter_without_dropping += 1
        else:
            iter_without_dropping = 0
            remove_order = np.argsort(z[ z < threshold ])
            elimination_order[ original_feature_indices[ z < threshold ][ remove_order ] ] = \
                np.arange(0, n_features_to_drop) + n_removed_features + 1
            n_removed_features += n_features_to_drop
            length -= n_features_to_drop
        
            # Delete from X, z and original_features the selected attributes 

            final_X = final_X[ :, z >= threshold ]
            original_feature_indices = original_feature_indices[ z >= threshold ]
            z = z[ z >= threshold ]

    # We remove all remaining features

    if length > 0:
            remove_order = np.argsort(z)
            elimination_order[ original_feature_indices[ remove_order ] ] = \
                np.arange(0, length) + n_removed_features + 1

    return np.argsort(-elimination_order)  # So array starts at 0 (python indexing)

