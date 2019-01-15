
# coding: utf-8

# In[9]:

import numpy as np

def nonlin(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

x = np.array([[0,0,1],
             [0,1,1],
             [1,0,1],
             [1,1,1]])

y = np.array([[0],
             [1],
             [1],
             [0]])

np.random.seed(1)

syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for j in xrange(60000):
    k0 = x
    k1 = nonlin(np.dot(k0,syn0))
    k2 = nonlin(np.dot(k1,syn1))
    
    k2_error = y - k2
    
    if(j % 10000) == 0:
        print("Error:"+str(np.mean(np.abs(k2_error))))
        
    k2_delta = k2_error*nonlin(k2,deriv=True)
    
    k1_error = k2_delta.dot(syn1.T)
    
    k1_delta = k1_error*nonlin(k1,deriv=True)
    
    syn1 += k1.T.dot(k2_delta)
    syn0 += k0.T.dot(k1_delta)


# In[5]:

print x


# In[6]:

print y


# In[7]:

print syn0


# In[8]:

print syn1


# In[ ]:



