from sklearn.model_selection import train_test_split
import torch

def calc_accuracy(out, label):
    predictions = torch.argmax(out, dim=1)

    correct_predictions = torch.eq(predictions, label).sum().item()
    accuracy = correct_predictions / label.size(0)
    return accuracy

def split_data(data, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    train_data, temp_data = train_test_split(data, test_size=1-train_ratio, random_state=42)
    valid_data, test_data = train_test_split(temp_data, test_size=test_ratio/(test_ratio + valid_ratio), random_state=42)
    return train_data, valid_data, test_data

