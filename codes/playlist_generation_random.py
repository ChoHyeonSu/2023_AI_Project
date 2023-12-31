import random
import pandas as pd
import numpy as np
from kl_divergence import kl_divergence_with_smoothing
from kl_divergence import str_to_list

def random_sampling(df, count=10):
    # 랜덤하게 N개의 index를 선택하고 제목도 list에 저장
    random_list = random.sample(range(0, len(df)), count)
    random_name = df.loc[random_list, 'song_name'].tolist()

    print('-'*30)
    print(f'Index: {random_list}')
    print(f'Song Name: {random_name}')

def label_sampling(df, index, count=10):
    # 샘플곡의 확률 분포로 모든 곡들에 대한 kl divergence 생성
    kl_divergence_total_list = []
    kl_divergence_label_list = []
    sample_prob_str = df.loc[index, 'probability']

    for i in range(len(df)):
        if i == index:
            kl_divergence_total_list.append('Nan')
            continue
        new_prob_str = df.loc[i, 'probability']

        sample_prob = str_to_list(sample_prob_str)
        new_prob = str_to_list(new_prob_str)
        
        kld = kl_divergence_with_smoothing(sample_prob, new_prob)
        kl_divergence_total_list.append(kld)

    # 샘플곡의 라벨과 동일한 곡 N개 추출
    sample_label = df.loc[index, 'Emotion']
    new_df = df[df['Emotion'] == sample_label].sample(count)

    label_list = new_df.index.values.tolist()
    label_name = new_df['song_name'].tolist()

    # 같은 라벨 곡들의 kl divergence를 보기 위한 코드
    for i in label_list:
        kl_divergence_label_list.append(kl_divergence_total_list[i])

    print('-'*30)
    print(f'Index: {label_list}')
    print(f'Song Name: {label_name}')
    print(f'KL Divergence: {kl_divergence_label_list}')

def baseline_sampling(df, df2, index, count=10):
    # 샘플곡의 확률 분포로 모든 곡들에 대한 kl divergence 생성
    kl_divergence_total_list = []
    kl_divergence_base_list = []
    sample_prob_str = df.loc[index, 'probability']

    for i in range(len(df2)):
        if i == index:
            kl_divergence_total_list.append('Nan')
            continue
        new_prob_str = df2.loc[i, 'probability']

        sample_prob = str_to_list(sample_prob_str)
        new_prob = list(map(float, new_prob_str[1:-1].split()))
        
        kld = kl_divergence_with_smoothing(sample_prob, new_prob)
        kl_divergence_total_list.append(kld)
    
    # kl divergence 값이 낮은 N개 추출
    base_list = np.argsort(kl_divergence_total_list)[:count].tolist()
    base_name = df['song_name'][base_list].tolist()

    # 같은 라벨 곡들의 kl divergence를 보기 위한 코드
    for i in base_list:
        kl_divergence_base_list.append(kl_divergence_total_list[i])

    print('-'*30)
    print(f'Index: {base_list}')
    print(f'Baseline Model Song Name: {base_name}')
    print(f'KL Divergence: {kl_divergence_base_list}')

def our_sampling(df, df2, index, count=10):
    # 샘플곡의 확률 분포로 모든 곡들에 대한 kl divergence 생성
    kl_divergence_total_list = []
    kl_divergence_our_list = []
    sample_prob_str = df.loc[index, 'probability']

    for i in range(len(df2)):
        if i == index:
            kl_divergence_total_list.append('Nan')
            continue
        new_prob_str = df2.loc[i, 'probability']

        sample_prob = str_to_list(sample_prob_str)
        new_prob = str_to_list(new_prob_str)
        
        kld = kl_divergence_with_smoothing(sample_prob, new_prob)
        kl_divergence_total_list.append(kld)
    
    # kl divergence 값이 낮은 N개 추출
    our_list = np.argsort(kl_divergence_total_list)[:count].tolist()
    our_name = df['song_name'][our_list].tolist()

    # 같은 라벨 곡들의 kl divergence를 보기 위한 코드
    for i in our_list:
        kl_divergence_our_list.append(kl_divergence_total_list[i])

    print('-'*30)
    print(f'Index: {our_list}')
    print(f'Our Model Song Name: {our_name}')
    print(f'KL Divergence: {kl_divergence_our_list}')

# 메인 코드
song_prob_gold_path = '/Users/jay/codes/classes/ai-project/team5-project/datasets/step2/song_prob.csv'
song_prob_baseline_path = '/Users/jay/codes/classes/ai-project/team5-project/datasets/step2/baseline_2304.csv'
song_prob_our_path = '/Users/jay/codes/classes/ai-project/team5-project/datasets/step2/our_2304.csv'

song_prob_gold = pd.read_csv(song_prob_gold_path)
song_prob_baseline = pd.read_csv(song_prob_baseline_path)
song_prob_our = pd.read_csv(song_prob_our_path)

# 노래 한 곡을 샘플링하는 코드
sample = random.randint(0, len(song_prob_gold))
sample_label = song_prob_gold.loc[sample, 'Emotion']
sample_name = song_prob_gold.loc[sample, 'song_name']

print(f'Sample Emotion: {sample_label}')
print(f'Sample Name: {sample_name}')
random_sampling(song_prob_gold)
label_sampling(song_prob_gold, sample)
baseline_sampling(song_prob_gold, song_prob_baseline, sample)
our_sampling(song_prob_gold, song_prob_our, sample)