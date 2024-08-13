import itertools
import math as math
import numpy as np
import sympy as sp

import matplotlib.pyplot as plt
import scipy.linalg as scln
from scipy.integrate import trapz
import itertools
from sympy.tensor.array.expressions import ArraySymbol
from sympy.abc import i, j, k
import tensorflow as tf

letters=["i","j","k","l","a","b","c","d","m","n","o","p","q","w","r","t"]



def get_path(N, max_up, n_up, n_diff, i, all_paths):
    if n_up == max_up:
        for j in range(i, len(N)):
            N[j] = "down"
        all_paths.append(N.copy())
        return
    elif len(N) - n_up == max_up:
        for j in range(i, len(N)):
            N[j] = "up"
        all_paths.append(N.copy())
        return
    if n_diff == 0:
        N[i] = "up"
        get_path(N, max_up, n_up+1, n_diff+1, i+1, all_paths)
    else:
        N[i] = "up"
        get_path(N, max_up, n_up+1, n_diff+1, i+1, all_paths)
        N[i] = "down"
        get_path(N, max_up, n_up, n_diff-1, i+1, all_paths)
        
def dycktocor(path,GG,I_val,I0_val):
    cl=0
    height=0
    set=[GG]
    setsym=[]
    x=0
    count=1
    indic="si"
    TI_val=I_val.copy()
    for i in range(4):
        for j in range(4):
            TI_val[i,j]+=-1
    JJ=GG @ GG
    for i in enumerate(path):   
        if i[1]=='1':
            cl+=int(i[1])
            height+=1
            count=1
        else:
            if count==1:
                #set.append(f"T^{height}_(x_{int(i[0]/2-height/2)}x_{int(i[0]/2+height/2)})")
                set.append(TI_val[:,:,height])
                setsym.append(f"T^{height}_(x_{int(i[0]/2-height/2)}x_{int(i[0]/2+height/2)})")
                ss=list(itertools.combinations(range(int(i[0]/2-height/2),int(height/2+i[0]/2+1)),2))
                x+=height
                #print(range(int(i[0]/2-height/2),int(height/2+i[0]/2+1)))
                #print(ss)
                indic=indic+f", {letters[int(i[0]/2-height/2)]}{letters[int(i[0]/2+height/2)]}"
                for j in ss:
                    k=j[1]-j[0]
                    if j[1]-j[0]<height:
                        if f"I^{k}_(x_{int(j[0])}x_{int(j[1])})" not in setsym:
                            setsym.append(f"I^{k}_(x_{int(j[0])}x_{int(j[1])})")
                            set.append(I_val[:,:,k])
                            indic=indic+f", {letters[int(j[0])]}{letters[int(j[1])]}"
                cl=0    
            height+=-1
            count=0
    for i in range(int(len(path)/2+1)):
        indic=indic+f", {letters[int(i)]}"
        set.append(I0_val)
    for i in range(int(len(path)/2)):
        indic=indic+f", {letters[int(i)]}{letters[int(i+1)]}"
        set.append(JJ)
    indic=indic+f", {letters[int(len(path)/2)]}e->se"
    set.append(GG)
    #set.append(x)
    set.append(indic)
    K=tf.einsum(set[-1], *set[:-1]) 
    #print(setsym)
    return(K)


def dycktocord(path,GG,I_val,I0_val):
    cl=0
    height=0
    set=[GG]
    setsym=[]
    x=0
    count=1
    indic="si"
    JJ=GG @ GG
    for i in enumerate(path):   
        if i[1]=='1':
            cl+=int(i[1])
            height+=1
            count=1
        else:
            if count==1:
                #set.append(f"T^{height}_(x_{int(i[0]/2-height/2)}x_{int(i[0]/2+height/2)})")
                #set.append(TI_val[:,:,height])
                #setsym.append(f"T^{height}_(x_{int(i[0]/2-height/2)}x_{int(i[0]/2+height/2)})")
                ss=list(itertools.combinations(range(int(i[0]/2-height/2),int(height/2+i[0]/2+1)),2))
                x+=height
                #print(range(int(i[0]/2-height/2),int(height/2+i[0]/2+1)))
                #print(ss)
                #indic=indic+f", {letters[int(i[0]/2-height/2)]}{letters[int(i[0]/2+height/2)]}"
                for j in ss:
                    k=j[1]-j[0]
                    if j[1]-j[0]<height:
                        if f"I^{k}_(x_{int(j[0])}x_{int(j[1])})" not in setsym:
                            setsym.append(f"I^{k}_(x_{int(j[0])}x_{int(j[1])})")
                            set.append(I_val[:,:,k])
                            indic=indic+f", {letters[int(j[0])]}{letters[int(j[1])]}"
                cl=0    
            height+=-1
            count=0
    for i in range(int(len(path)/2+1)):
        indic=indic+f", {letters[int(i)]}"
        set.append(I0_val)
        setsym.append('I0_val')
    for i in range(int(len(path)/2)):
        indic=indic+f", {letters[int(i)]}{letters[int(i+1)]}"
        set.append(JJ)
        setsym.append('JJ')
    indic=indic+f", {letters[int(len(path)/2)]}e->se"
    set.append(GG)
    #set.append(x)
    set.append(indic)
    #print(setsym)
    K=tf.einsum(set[-1], *set[:-1]) 
    return(np.array(K))


def wordgenerate(memorylength):
    for max_up in [memorylength]:
        all_paths = []
        dyckwords=[]
        # max_up = 8
        get_path(["" for i in range(2*max_up)], max_up, 0, 0, 0, all_paths)
        #print(f"max_up {max_up}, num_path {len(all_paths)}")
        for path in all_paths:
            print_path = []
            for element in path:
                if element == "up":
                    print_path.append("1")
                else:
                    print_path.append("0")
            dyckwords.append(print_path)
        return dyckwords