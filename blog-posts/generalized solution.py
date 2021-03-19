# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 18:27:47 2021

@author: maste
"""

import numpy as np
import matplotlib.pyplot as plt

'''
Formalization of the Rescorla-Wagner learning algorithm into
linear algebra notation. Allows for expansion of the cue and field.

alpha - float, 0<x<=1, learning rate associated with an element within the cue
beta - float, 0<x<=1, general learning rate

lambda=r - right now, float, expected output in response field. TODO generalize
V - float matrix, dims:[len(cue), len(response)], weight matrix
c - vector, floats, cue vector

Update rule:
    V_t+1 = V_t + c^T * Delta(V)
    Delta(V) = alpha * beta * (lambda - c @ V)

'''

labels = ['A_+','AB_o','AC_+','ABC_o']

c = np.array([[1,0,0],
              [1,1,0],
              [1,0,1],
              [1,1,1]])
r = np.array([[1],
              [0],
              [1],
              [0]])

V = np.ones((c.shape[1],r.shape[1])) * .1

alpha = 1
beta = .2

def rw_update(V,c,r,alpha,beta):
    return V + c.T @ (alpha * beta * (r - (c @ V)))

vs = []
for i in range(50):
    # shuffle(train_c_r)
    for i in np.arange(c.shape[0]):
        V = rw_update(V, c[i:i+1,:], r[i:i+1,:], alpha, beta)
    vs.append(V)

vs = np.hstack(vs)

# get the activations for each of the associations in c_rs
acts = c @ vs

x = np.arange(acts.shape[1])
plt.figure(num = 3, figsize=(8, 5))
for ind in np.arange(acts.shape[0]):
    plt.plot(x,acts[ind].T,label=labels[ind])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


##############
# 2 discriminators

V_1 = np.ones((3,1)) * .1
V_2 = np.ones((3,1)) * .1


vs_1 = []
vs_2 = []
for i in range(50):

    for i in np.arange(c.shape[0]):
        if i < 2:
            V_1 = rw_update(V_1, c[i:i+1,:], r[i:i+1,:], alpha, beta)
        else:
            V_2 = rw_update(V_2, c[i:i+1,:], r[i:i+1,:], alpha, beta)

    vs_1.append(V_1)
    vs_2.append(V_2)

vs_1 = np.hstack(vs_1)
vs_2 = np.hstack(vs_2)

acts = []
acts.append(c[:2] @ vs_1)
acts.append(c[2:] @ vs_2)
acts = np.vstack(acts)

x = np.arange(acts.shape[1])
plt.figure(num = 3, figsize=(8, 5))
for ind in np.arange(acts.shape[0]):
    plt.plot(x,acts[ind].T,label=labels[ind])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


###########################################################
# Project cue field into an arbitrarly high dimensional random vector space

# projection matrix - 3 orthogonal vectors
d = 50
projection = np.random.normal(loc = 0, scale = 1/np.sqrt(d), size = (c.shape[1], d))
c_ = c @ projection


V_1 = np.ones((d,1)) * .1
V_2 = np.ones((d,1)) * .1


vs_1 = []
vs_2 = []
for i in range(50):

    for i in np.arange(c.shape[0]):
        if i < 2:
            V_1 = rw_update(V_1, c_[i:i+1,:], r[i:i+1,:], alpha, beta)
        else:
            V_2 = rw_update(V_2, c_[i:i+1,:], r[i:i+1,:], alpha, beta)

    vs_1.append(V_1)
    vs_2.append(V_2)

vs_1 = np.hstack(vs_1)
vs_2 = np.hstack(vs_2)

acts = []
acts.append(c_[:2] @ vs_1)
acts.append(c_[2:] @ vs_2)
acts = np.vstack(acts)

x = np.arange(acts.shape[1])
plt.figure(num = 3, figsize=(8, 5))
for ind in np.arange(acts.shape[0]):
    plt.plot(x,acts[ind].T,label=labels[ind])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

###########################################################
# Project response field into an arbitrarly high dimensional random vector space

# projection matrix - 3 orthogonal vectors
d = 50
projection = np.random.normal(loc = 0, scale = 1/np.sqrt(d), size = (r.shape[1], d))
r_ = r @ projection


V_1 = np.ones((3,d)) * .1
V_2 = np.ones((3,d)) * .1


vs_1 = []
vs_2 = []
for i in range(50):

    for i in np.arange(c.shape[0]):
        if i < 2:
            V_1 = rw_update(V_1, c[i:i+1,:], r_[i:i+1,:], alpha, beta)
        else:
            V_2 = rw_update(V_2, c[i:i+1,:], r_[i:i+1,:], alpha, beta)

    vs_1.append(V_1[:,:,np.newaxis])
    vs_2.append(V_2[:,:,np.newaxis])

vs_1 = np.concatenate(vs_1, axis=-1)
vs_2 = np.concatenate(vs_2, axis=-1)

acts = []
acts.append(np.tensordot(c[:2], vs_1, axes=([1],[0])))
acts.append(np.tensordot(c[2:], vs_2, axes=([1],[0])))
acts = np.concatenate(acts, axis=0)

out = np.sqrt((np.subtract(acts, projection[:,:,np.newaxis])**2).sum(1))
x = np.arange(out.shape[1])
plt.figure(num = 3, figsize=(8, 5))
for ind in np.arange(out.shape[0]):
    plt.plot(x,out[ind].T,label=labels[ind])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


###########################################################
# Project both cue and response field into an arbitrarly high dimensional random vector space

# projection matrix - 3 orthogonal vectors
d_r = 50
d_c = 50
projection_r = np.random.normal(loc = 0, scale = 1/np.sqrt(d), size = (r.shape[1], d))
projection_c = np.random.normal(loc = 0, scale = 1/np.sqrt(d), size = (c.shape[1], d))
r_ = r @ projection_r
c_ = c @ projection_c


V_1 = np.ones((d_c,d_r)) * .1
V_2 = np.ones((d_c,d_r)) * .1


vs_1 = []
vs_2 = []
for i in range(50):

    for i in np.arange(c.shape[0]):
        if i < 2:
            V_1 = rw_update(V_1, c_[i:i+1,:], r_[i:i+1,:], alpha, beta)
        else:
            V_2 = rw_update(V_2, c_[i:i+1,:], r_[i:i+1,:], alpha, beta)

    vs_1.append(V_1[:,:,np.newaxis])
    vs_2.append(V_2[:,:,np.newaxis])

vs_1 = np.concatenate(vs_1, axis=-1)
vs_2 = np.concatenate(vs_2, axis=-1)

acts = []
acts.append(np.tensordot(c_[:2], vs_1, axes=([1],[0])))
acts.append(np.tensordot(c_[2:], vs_2, axes=([1],[0])))
acts = np.concatenate(acts, axis=0)

out = np.sqrt((np.subtract(acts, projection_r[:,:,np.newaxis])**2).sum(1))
x = np.arange(out.shape[1])
plt.figure(num = 3, figsize=(8, 5))
for ind in np.arange(out.shape[0]):
    plt.plot(x,out[ind].T,label=labels[ind])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
