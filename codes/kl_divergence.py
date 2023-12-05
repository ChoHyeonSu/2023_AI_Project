import pandas as pd
from scipy.stats import entropy
import ast
import numpy as np

def str_to_list(str_list):
    return ast.literal_eval(str_list)

def kl_divergence_with_smoothing(p, q, smoothing_value=1e-9):
    p = np.array(p)
    q = np.array(q)
    
    p += smoothing_value
    q += smoothing_value
    
    p /= p.sum()
    q /= q.sum()
    
    return entropy(p, q)