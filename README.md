# PU_learning
 This is repository for implementation of Positive-Unlabeled Learning algorithm. The two variations of this algorithm were described in paper ElkanKDD08.pdf in 2 and 3 paragraph. 
 
 The main idea is that we have two sets of examples: positive and negative. But unlike the usual classificator we have incomplete information. Namely, we have positive and unlabeled data. Some of unlabeled examples are positive and some are negative. The goal is to make the classifier which will separate positive data from negative data.
 
 * 1Using_traditional_classifier.py -- Python 3 file with implementation of PU-model using traditional classifier on non-traditional input. Its output is an illustration of the method on data sampled from gaussians.
 
 * 2Weighting_unlabeled_examples.py -- Python 3 file with implementation of PU-model using weighted unlabeled examples. Its output is an illustration of the method on data sampled from gaussians.
 
 * Comparison.py -- Python 3 file with implementation of both methods to compare them on single dataset sampled from gaussians. 

 * Illustrations.pdf -- file with illustrations of results and comparison of two methods listed above.
