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

def baseline_sampling(df, df2, prob_list, count=10):
    # 샘플곡의 확률 분포로 모든 곡들에 대한 kl divergence 생성
    kl_divergence_total_list = []
    kl_divergence_base_list = []
    sample_prob = prob_list

    for i in range(len(df2)):
        new_prob_str = df2.loc[i, 'probability']
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

def our_sampling(df, df2, prob_list, count=10):
    # 샘플곡의 확률 분포로 모든 곡들에 대한 kl divergence 생성
    kl_divergence_total_list = []
    kl_divergence_our_list = []
    sample_prob = prob_list

    for i in range(len(df2)):
        new_prob_str = df2.loc[i, 'probability']
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

# 상황 및 감정 분포 리스트
situation_list = ['시험 준비할 때', '좋아하는 사람과 첫 데이트할 때', '새로운 도시로 이사할 때', '친한 친구의 결혼식에 참석할 때', '이별을 겪고 있을 때', '전 여자친구가 생각날 때', '중요한 소식을 기다리고 있을 때', '전 여자친구를 길에서 만났을 때', '연말에 듣는 노래', '일할 때 듣는 노래']
ratio_list = [[5, 0, 10, 75, 5, 5], [10, 30, 50, 5, 0, 5], [10, 0, 20, 30, 30, 10], [0, 70, 20, 0, 0, 10], [30, 0, 0, 10, 30, 30], [30, 0, 0, 0, 20, 50], [10, 0, 40, 40, 5, 5], [20, 10, 15, 30, 15, 10], [5, 60, 30, 5, 0, 0], [0, 40, 35, 15, 0, 10]]
prob_list = (np.array(ratio_list) / 100)

# 상황 하나를 고르는 과정
sample_index = int(input('0 ~ 9까지 중에 입력하시오\n'))

random_sampling(song_prob_gold)
baseline_sampling(song_prob_gold, song_prob_baseline, prob_list[sample_index])
our_sampling(song_prob_gold, song_prob_our, prob_list[sample_index])