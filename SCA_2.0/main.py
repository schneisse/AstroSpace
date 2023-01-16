#!/usr/bin/env python
# coding: utf-8

# In[1]:


from point_cloud import *
from utils import RotVect, Segm, SumDirection
import branch as br
from parameters import R, Ar, Dk, Bl, N, leaves, tree

from copy import copy
from scipy.spatial import KDTree
import numpy as np
import matplotlib.pyplot as plt

# default parameters
mm = 1
mkm = 0.001*mm
nm = 0.001*mkm

initial_br = br.Branch(np.array([0, 0]), np.array([0, 1]), Bl)


# In[ ]:





# # Main code

# In[4]:


ROOTs = [copy(initial_br)]
lvs = leaves.leaves
output = [copy(initial_br)]
borders = False

while borders == False:

    if len(ROOTs) > 1:
        for n, r in enumerate(ROOTs):
            new_br = br.Grow2(lvs, r, Ar, Dk, Bl)
            borders = br.borders_reached(new_br)
            
            if br.match(ROOTs, new_br) == True:
                continue
            elif br.prob(n, ROOTs, v=1) == False:
                continue
            else:
                new_lvs = br.kill_attr(lvs, new_br)
                lvs = new_lvs
                tree = KDTree(lvs)
                # ROOTs[n] = new_br
                output.append(new_br)
                ROOTs.append(new_br)

    else: 
        new_br = br.Grow2(lvs, ROOTs[0], Ar, Dk, Bl)
        new_lvs = br.kill_attr(lvs, new_br)
        lvs = new_lvs
        tree = KDTree(lvs)
        ROOTs.append(new_br)
        # output.append(new_br)
        borders = br.borders_reached(new_br)


# In[8]:


output = np.array(list(map(lambda x: list(x.segment), output)))


# In[ ]:





# In[ ]:




