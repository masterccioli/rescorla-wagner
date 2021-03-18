# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 12:34:41 2021

@author: maste
"""

import numpy as np
from random import shuffle
import matplotlib.pyplot as plt

def rw_update(V,c_r,alpha,beta):
    return V+(c_r[0]*alpha*beta*(c_r[1]-(c_r[0]*V).sum()))

# initialize association matrix, {(A,B), (1/0)}
V = np.ones((3,1)) * .1
alpha = 1
beta = .2

labels = ['A_+','AB_o','AC_+','ABC_o']

c_rs = [(np.array([1,0,0]).reshape((3,1)), np.array([[1]])), # A_+
       (np.array([1,1,0]).reshape((3,1)), np.array([[0]])), # AB_o
        (np.array([1,0,1]).reshape((3,1)), np.array([[1]])), # AC_+
        (np.array([1,1,1]).reshape((3,1)), np.array([[0]]))] # ABC_o

# train_c_r = c_rs.copy()

vs = []
for i in range(10):
    # shuffle(train_c_r)
    for c_r in c_rs:
        V = rw_update(V, c_r, alpha, beta)
    vs.append(V)

vs = np.hstack(vs)

# get the activations for each of the associations in c_rs
acts = []
for c in c_rs:
    act = []
    for v in vs.T:
        act.append((c[0].T*v).sum())

    acts.append(np.hstack(act))
acts = np.vstack(acts)

x = np.arange(acts.shape[1])
plt.figure(num = 3, figsize=(8, 5))
for ind,i in enumerate(acts):
    plt.plot(x,i,label=labels[ind])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)




V_1 = np.ones((3,1)) * .1
V_2 = np.ones((3,1)) * .1
V_3 = np.ones((3,1)) * .1
V_4 = np.ones((3,1)) * .1
alpha = 1
beta = .2

c_rs = [(np.array([1,0,0]).reshape((3,1)), np.array([[1]])), # A_+
       (np.array([1,1,0]).reshape((3,1)), np.array([[0]])), # AB_o
        (np.array([1,0,1]).reshape((3,1)), np.array([[1]])), # AC_+
        (np.array([1,1,1]).reshape((3,1)), np.array([[0]]))] # ABC_o

# train_c_r = c_rs.copy()

vs_1 = []
vs_2 = []
vs_3 = []
vs_4 = []
for i in range(15):
    # shuffle(train_c_r)
    # for c_r in c_rs:
    #     V = rw_update(V, c_r, alpha, beta)
    #     vs.append(V)

    c_r = c_rs[0]
    V_1 = rw_update(V_1, c_r, alpha, beta)
    vs_1.append(V_1)
    c_r = c_rs[1]
    V_2 = rw_update(V_2, c_r, alpha, beta)
    vs_2.append(V_2)
    c_r = c_rs[2]
    V_3 = rw_update(V_3, c_r, alpha, beta)
    vs_3.append(V_3)
    c_r = c_rs[3]
    V_4 = rw_update(V_4, c_r, alpha, beta)
    vs_4.append(V_4)

vs_1 = np.hstack(vs_1)
vs_2 = np.hstack(vs_2)
vs_3 = np.hstack(vs_3)
vs_4 = np.hstack(vs_4)

# get the activations for each of the associations in c_rs
acts = []

act = []
c = c_rs[0]
for v in vs_1.T:
    act.append((c[0].T*v).sum())
acts.append(np.hstack(act))

act = []
c = c_rs[1]
for v in vs_2.T:
    act.append((c[0].T*v).sum())
acts.append(np.hstack(act))

act = []
c = c_rs[2]
for v in vs_3.T:
    act.append((c[0].T*v).sum())
acts.append(np.hstack(act))

act = []
c = c_rs[3]
for v in vs_4.T:
    act.append((c[0].T*v).sum())
acts.append(np.hstack(act))

acts = np.vstack(acts)

x = np.arange(acts.shape[1])
plt.figure(num = 3, figsize=(8, 5))
for ind,i in enumerate(acts):
    plt.plot(x,i,label=labels[ind])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)














V_1 = np.ones((3,1)) * .1
V_2 = np.ones((3,1)) * .1

alpha = 1
beta = .2

c_rs = [(np.array([1,0,0]).reshape((3,1)), np.array([[1]])), # A_+
       (np.array([1,1,0]).reshape((3,1)), np.array([[0]])), # AB_o
        (np.array([1,0,1]).reshape((3,1)), np.array([[1]])), # AC_+
        (np.array([1,1,1]).reshape((3,1)), np.array([[0]]))] # ABC_o

# train_c_r = c_rs.copy()

vs_1 = []
vs_2 = []
for i in range(15):
    # shuffle(train_c_r)
    # for c_r in c_rs:
    #     V = rw_update(V, c_r, alpha, beta)
    #     vs.append(V)

    c_r = c_rs[0]
    V_1 = rw_update(V_1, c_r, alpha, beta)
    c_r = c_rs[1]
    V_1 = rw_update(V_1, c_r, alpha, beta)
    vs_1.append(V_1)

    c_r = c_rs[2]
    V_2 = rw_update(V_2, c_r, alpha, beta)
    c_r = c_rs[3]
    V_2 = rw_update(V_2, c_r, alpha, beta)
    vs_2.append(V_2)

vs_1 = np.hstack(vs_1)
vs_2 = np.hstack(vs_2)

# get the activations for each of the associations in c_rs
acts = []

act = []
c = c_rs[0]
for v in vs_1.T:
    act.append((c[0].T*v).sum())
acts.append(np.hstack(act))

act = []
c = c_rs[1]
for v in vs_1.T:
    act.append((c[0].T*v).sum())
acts.append(np.hstack(act))

act = []
c = c_rs[2]
for v in vs_2.T:
    act.append((c[0].T*v).sum())
acts.append(np.hstack(act))

act = []
c = c_rs[3]
for v in vs_2.T:
    act.append((c[0].T*v).sum())
acts.append(np.hstack(act))

acts = np.vstack(acts)

x = np.arange(acts.shape[1])
plt.figure(num = 3, figsize=(8, 5))
for ind,i in enumerate(acts):
    plt.plot(x,i,label=labels[ind])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
