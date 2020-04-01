#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:05:37 2020

@author: maria

Positive-Unlabeled Learning Using Weighted Unlabeled Examples
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import  linear_model
from scipy import stats



def sample_point(centers, cov):
    i = np.random.randint(len(centers))
    return np.random.multivariate_normal(centers[i], cov)

def create_set_gauss(var = 0.2, num_of_centers = 20, prob_of_labeled = 0.3):
    """
    Returns 2-D array of points and their labels. Points are sampled from 
    'num_of_centers' gaussians with covariance matrix 'var' * np.identity(1)
    prob_of labeled -- int, probability that point is labeled provided that 
    it is positive
    """
    mean0 = np.random.multivariate_normal([-1, -1], 0.1 * np.identity(2), 
                                          size = num_of_centers // 2)
    mean1 = np.random.multivariate_normal([1, 1], 0.1 * np.identity(2), 
                                          size = num_of_centers // 2)

    sample_points0 = np.array([sample_point(mean0, var * np.identity(2)) for _ in range(100)])
    sample_points1 = np.array([sample_point(mean1, var * np.identity(2)) for _ in range(200)])

    
    return np.array([(p, 0) for p in sample_points0] +
                       [(p, stats.bernoulli.rvs(prob_of_labeled)) for p in sample_points1])


def labeled(points):
    return [p[0] for p in points if p[1] == 1]

def unlabeled(points):
    return [p[0] for p in points if p[1] == 0]

def draw_picture(points):
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
    plt.plot(unlab_x, unlab_y, 'o')
    plt.plot(lab_x, lab_y, 'o')
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
        
    
    #Draw separating line
    x, y = np.mgrid[min_x : max_x :.01, min_y : max_y :.01]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    pos_lin = pos.reshape(pos.shape[0] * pos.shape[1], 2)
    pred_lin = new_classifier.predict_proba(pos_lin)[:,1]
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

#Create weighted data
new_data = []
new_labels = []
weights = []
probabilities = traditional_classifier.predict_proba(data)
for i in range(len(points)):
    if points[i][1] == 1:
        new_data.append(points[i][0])
        new_labels.append(points[i][1])
        weights.append(1)
    else:
        prob_labeled = probabilities[i][1]
        weight = ((1 - estimator) * prob_labeled)/ (estimator * (1 - prob_labeled))
        new_data.append(points[i][0])
        new_labels.append(1)
        weights.append(weight)
        
        new_data.append(points[i][0])
        new_labels.append(0)
        weights.append(1 - weight)

#Train on new data        
new_classifier = linear_model.LogisticRegression()
new_classifier.fit(new_data, new_labels, weights)





draw_picture(points)






