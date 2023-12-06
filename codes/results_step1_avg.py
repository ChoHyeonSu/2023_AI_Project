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

avg_our = np.sum(our_kl) / len(our_kl)
avg_baseline1 = np.sum(baseline1_kl) / len(baseline1_kl)
avg_baseline2 = np.sum(baseline2_kl) / len(baseline2_kl)

print('Average KL Divergence values')
print(f'Ours: {avg_our}')
print(f'Baseline1: {avg_baseline1}')
print(f'Baseline2: {avg_baseline2}')