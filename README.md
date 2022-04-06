# PU_learning
 This is a repository for an implementation of the Positive-Unlabeled Learning algorithm. The two variations of this algorithm were described in the paper ElkanKDD08.pdf in 2nd and 3rd paragraph. 
 
 The main idea isthe following. Assume we have two sets of examples: positive and negative ones. But unlike the usual classificator we have incomplete information. Namely, there is only positive and unlabeled data. Some of unlabeled examples are positive and some are negative. The goal is to make a classifier which will separate positive data from negative data.
 
 * 1Using_traditional_classifier.py is a Python 3 file with the implementation of PU-model using a traditional classifier on non-traditional input. Its output is an illustration of the method on the data sampled from gaussians.
 
 * 2Weighting_unlabeled_examples.py is a Python 3 file with the implementation of PU-model using weighted unlabeled examples. Its output is an illustration of the method on the data sampled from gaussians.
 
 * Comparison.py is a Python 3 file with the implementation of both methods in order to compare them on a single dataset sampled from gaussians. 

 * Illustrations.pdf is a file with the illustrations of the results and comparison of two methods listed above.
