import random
import pandas as pd
import numpy as np
from kl_divergence import kl_divergence_with_smoothing
from kl_divergence import str_to_list

# test_path = '/Users/jay/codes/classes/ai-project/team5-project/datasets/results/test.csv'
test_path = '/Users/jay/codes/classes/ai-project/team5-project/datasets/results/validation.csv'
test = pd.read_csv(test_path)

our_kl = []
baseline1_kl = []
baseline2_kl = []

for i in range(len(test)):
    golden_dist_str = test.loc[i, 'golden_dist']
    golden_dist = str_to_list(golden_dist_str)

    baseline1_dist_str = test.loc[i, 'base1_dist']
    baseline1_dist = list(map(float, baseline1_dist_str[1:-1].split()))

    baseline2_dist_str = test.loc[i, 'base2_dist']
    baseline2_dist = list(map(float, baseline2_dist_str[1:-1].split()))

    our_dist_str = test.loc[i, 'ours_dist']
    our_dist = str_to_list(our_dist_str)

    our_kl.append(kl_divergence_with_smoothing(golden_dist, our_dist))
    baseline1_kl.append(kl_divergence_with_smoothing(golden_dist, baseline1_dist))
    baseline2_kl.append(kl_divergence_with_smoothing(golden_dist, baseline2_dist))

count = [0, 0, 0]

for i in range(len(our_kl)):
    min = our_kl[i]
    if min < baseline1_kl[i]:
        min = baseline1_kl[i]
    if min < baseline2_kl[i]:
        min = baseline2_kl[i]
    
    if min == our_kl[i]:
        count[0] += 1
    if min == baseline1_kl[i]:
        count[1] += 1
    if min == baseline2_kl[i]:
        count[2] += 1

print('Minimum kl divergence value counts')
print(f'Ours: {count[0]}')
print(f'Baseline1: {count[1]}')
print(f'Baseline2: {count[2]}')