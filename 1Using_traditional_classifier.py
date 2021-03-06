#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:05:37 2020

@author: maria

Positive-Unlabeled Learning Using a Traditional Classifier from Nontraditional Input
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import  linear_model
from scipy import stats

   
    
def new_predict_proba(samples, estimator):
    """
    Arg: 
     samples -- 1-D array of samples
     
    Returns: 
     2-D array of probabilities of belonging to 2 classes in PU-model
    """
    t = traditional_classifier.predict_proba(samples)
    return np.array([(1 - t[i][1] / estimator, t[i][1] / estimator) for 
                     i in range(len(samples)) ])
    
    
def new_predict(samples, estimator):
    """
    Arg: 
     samples -- 1-D array of  samples
     
    Returns: 
     1-D array -- number of predicted group in PU-model
    """
    pr = new_predict_proba(samples, estimator)
    return np.array([0 if pr[i][0] > 0.5 else 1 for i in range(len(samples))])
    


def sample_point(centers, cov):
    i = np.random.randint(len(centers))
    return np.random.multivariate_normal(centers[i], cov)

def create_set_gauss(var = 0.1, num_of_centers = 20, prob_of_labeled = 0.3):
    """
    Returns 2-D array of points and their labels. Points are sampled from 
    'num_of_centers' gaussians with covariance matrix 'var' * np.identity(1)
    prob_of labeled -- int, probability that point is labeled provided that 
    it is positive
    """
    mean0 = np.random.multivariate_normal([-1, -1], 0.5 * np.identity(2), 
                                          size = num_of_centers // 2)
    mean1 = np.random.multivariate_normal([1, 1], 0.5 * np.identity(2), 
                                          size = num_of_centers // 2)

    sample_points0 = np.array([sample_point(mean0, var * np.identity(2)) for _ in range(100)])
    sample_points1 = np.array([sample_point(mean1, var * np.identity(2)) for _ in range(200)])

    
    return np.array([(p, 0) for p in sample_points0] +
                       [(p, stats.bernoulli.rvs(prob_of_labeled)) for p in sample_points1])


def labeled(points):
    return [p[0] for p in points if p[1] == 1]

def unlabeled(points):
    return [p[0] for p in points if p[1] == 0]

def draw_picture(points, estimator):
    """
    Arg: 
     points -- 2-D array of points and their labels
     
    Draws picture of data and separating hyperplane
    """

    
    unlab_x, unlab_y = [p[0] for p in unlabeled(points)], [p[1] for p in unlabeled(points)]
    lab_x, lab_y = [p[0] for p in labeled(points)], [p[1] for p in labeled(points)]
    
    min_x, max_x = min(unlab_x + lab_x), max(unlab_x + lab_x)
    min_y, max_y = min(unlab_y + lab_y), max(unlab_y + lab_y)
    
    #Draw  sample points
    plt.plot(unlab_x, unlab_y, 'o', label = 'unlabeled')
    plt.plot(lab_x, lab_y, 'o', label = 'labeled')
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.legend()
        
    
    #Draw separating line
    x, y = np.mgrid[min_x : max_x :.01, min_y : max_y :.01]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    pos_lin = pos.reshape(pos.shape[0] * pos.shape[1], 2)
    pred_lin = new_predict_proba(pos_lin, estimator)[:,1]
    pred = pred_lin.reshape(pos.shape[0], pos.shape[1])
    plt.contour(x, y, pred, levels=[0.5])
    

  
#Create sample set of labeled and unlabeled points 
points = create_set_gauss()
data = [points[i][0] for i in range(len(points))]
labels =[points[i][1] for i in range(len(points))]

#Train traditional logistic regression model
traditional_classifier = linear_model.LogisticRegression()
traditional_classifier.fit(data, labels)

#Estimate probability that point is labeled provided that it is positive
t = traditional_classifier.predict_proba(labeled(points))
estimator = sum([t[i][1] for i in range(len(labeled(points)))]) / len(labeled(points))

draw_picture(points, estimator)






