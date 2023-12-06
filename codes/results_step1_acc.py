import random
import pandas as pd
import numpy as np
from kl_divergence import kl_divergence_with_smoothing
from kl_divergence import str_to_list

# test_path = '/Users/jay/codes/classes/ai-project/team5-project/datasets/results/test.csv'
test_path = '/Users/jay/codes/classes/ai-project/team5-project/datasets/results/validation.csv'
test = pd.read_csv(test_path)

# 문자열로 된 감정 라벨을 정수로 매핑
def conversion(x):
    if x == '그리움':
        return 0
    if x == '사랑&기쁨':
        return 1
    if x == '설렘&심쿵':
        return 2
    if x == '스트레스&짜증':
        return 3
    if x == '외로울때':
        return 4
    if x == '슬픔':
        return 5

test['golden_em_label'] = test['golden_label'].apply(conversion)

# 각 모델의 distribution의 max와 golden label 비교
total_count = [0, 0, 0]

for i in range(len(test)):
    golden_index = test.loc[i, 'golden_em_label']

    baseline1_dist_str = test.loc[i, 'base1_dist']
    baseline1_dist = list(map(float, baseline1_dist_str[1:-1].split()))

    baseline2_dist_str = test.loc[i, 'base2_dist']
    baseline2_dist = list(map(float, baseline2_dist_str[1:-1].split()))

    our_dist_str = test.loc[i, 'ours_dist']
    our_dist = str_to_list(our_dist_str)

    if golden_index == np.argmax(our_dist):
        total_count[0] += 1

    if golden_index == np.argmax(baseline1_dist):
        total_count[1] += 1

    if golden_index == np.argmax(baseline2_dist):
        total_count[2] += 1

print('Accuracy')
print(f'Ours: {np.sum(total_count[0])/len(test)}')
print(f'Baseline1: {np.sum(total_count[1])/len(test)}')
print(f'Baseline2: {np.sum(total_count[2])/len(test)}')