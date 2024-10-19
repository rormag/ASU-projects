# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:44:08 2024

@author: rorym
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns                            # data visualization

heart_df = pd.read_csv('heart1.csv')
    


###############
##CORRELATION##
###############


# create the correlation
# take the absolute value since large negative are as useful as large positive
corr = heart_df.corr().abs()
#print(corr)


# set the correlations on the diagonal or lower triangle to zero,
# so they will not be reported as the highest ones.
# (The diagonal is always 1; the matrix is symmetric about the diagonal.)

# We clear the diagonal since the correlation with itself is always 1.

# Note the * in front of the argument in tri. That's because shape returns
# a tuple and * unrolls it so they become separate arguments.
#print(corr.values.shape)

# Note this will be element by element multiplication
corr *= np.tri(*corr.values.shape, k=-1).T
#print(corr)
#input()

# now unstack it so we can sort things
# note that zeros indicate no correlation OR that we cleared below the
# diagonal. Note that corr_unstack is a pandas series.
corr_unstack = corr.unstack()
#print(corr_unstack)
#print(type(corr_unstack))
#input()


corr_unstack = corr_unstack.copy()
# Sort values in descending order
corr_unstack.sort_values(inplace=True,ascending=False)
#print(corr_unstack)


# Now just print the top values
print("Top correlated variables")
print(corr_unstack.head(10))
print()


# Get the correlations with type
#with_type = corr_unstack.get(key="type")
#print(with_type)

print("Top variables correlated with a1p2")
print(corr_unstack.head(10)['a1p2'])
print()







##############
##COVARIANCE##
##############


# create the covariance
# take the absolute value since large negative are as useful as large positive
cov = heart_df.cov().abs()
print("Covariance Matrix")
print(cov)


# set the covariance on the diagonal or lower triangle to zero,
# so they will not be reported as the highest ones.
# (The diagonal is always 1; the matrix is symmetric about the diagonal.)

# We clear the diagonal since the covariance with itself is always 1.

# Note the * in front of the argument in tri. That's because shape returns
# a tuple and * unrolls it so they become separate arguments.
#print(cov.values.shape)

# Note this will be element by element multiplication
cov *= np.tri(*cov.values.shape, k=-1).T
#print(cov)


# now unstack it so we can sort things
# note that zeros indicate no covariance OR that we cleared below the
# diagonal. Note that cov_unstack is a pandas series.
cov_unstack = cov.unstack()
#print(cov_unstack)
#print(type(cov_unstack))


# Sort values in descending order
cov_unstack = cov_unstack.copy()
cov_unstack.sort_values(inplace=True,ascending=False)
print(cov_unstack)
print()

# Now just print the top values
print("Top covarying variables")
print(cov_unstack.head(10))
print()

print("Top covarying variables")
print(cov_unstack['a1p2'])
print()

# Get the covariance with type
with_type = cov_unstack.get(key="type")
print(with_type)

#PAIRPLOT


sns.set(style='whitegrid', context='notebook')   # set the apearance
sns.pairplot(heart_df,height=1.5)                # create the pair plots
plt.show()                                       # and show them





