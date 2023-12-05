import torch
from torch.utils.data import DataLoader
from model import BERTClassifier, BERTDataset 
from utils import calc_accuracy

def test_model(model, test_dataset, device):
    # 모델 로드
    model.to(device)

    # 데이터 로더 설정
    test_dataloader = DataLoader(test_dataset, batch_size=64, num_workers=5)

    # 성능 평가
    model.eval()
    total_acc, total_count = 0, 0
    with torch.no_grad():
        for token_ids, valid_length, segment_ids, label in test_dataloader:
            token_ids, segment_ids, label = token_ids.long().to(device), segment_ids.long().to(device), label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            total_acc += calc_accuracy(out, label)
            total_count += label.size(0)

    print(f'Test Accuracy: {total_acc / total_count:.4f}')
