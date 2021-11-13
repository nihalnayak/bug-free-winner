
## Taken calc_domainAveraged_FOSCTTM from SCOT

import numpy as np


def softmax(x):
	p_x = np.exp(x)
	return p_x/np.sum(p_x,axis=0)


def calc_domainAveraged_FOSCTTM(x1_mat, x2_mat):
	"""
	Outputs average FOSCTTM measure (averaged over both domains)
	Get the fraction matched for all data points in both directions
	Averages the fractions in both directions for each data point
	"""
	fracs1,xs = calc_frac_idx(x1_mat, x2_mat)
	fracs2,xs = calc_frac_idx(x2_mat, x1_mat)
	fracs = []
	for i in range(len(fracs1)):
		fracs.append((fracs1[i]+fracs2[i])/2)  
	return fracs


def calc_match(x1,x2):
	"""
    Output the matches probaility of the output coupling with truth align
    x1: output from scot
    x2: truth alignment matrix
	"""
	## Normalized the rows of output matrix x1 from scot 
	x1_norm= softmax(x1)
	x2 = x2.X.toarray()
	score=0
	for i in range(0,x2.shape[0]):
		score = score + np.dot(x2[i,np.nonzero(x2[i])].flatten(),x1_norm[i,np.nonzero(x2[i])].flatten())

	return score/x2.shape[0]