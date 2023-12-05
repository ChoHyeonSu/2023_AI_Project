import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from model import BERTClassifier, BERTDataset
from transformers import BertTokenizer
from train import train_model
#KoBERT
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
import gluonnlp as nlp

def get_data():
    data = pd.read_csv('dataset/diary_undersampling.csv')
    data_list = []
    for q, label in zip(data['diary'], data['emotion'])  :
        d = []
        d.append(q)
        d.append(str(label))

        data_list.append(d)

    dataset_train, dataset_test = train_test_split(data_list, test_size=0.1, stratify=data['emotion'], random_state=0)

    return dataset_train, dataset_test

def main():
    bertmodel, vocab = get_pytorch_kobert_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    max_len = 64

    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    train, test = get_data()
        
    data_train = BERTDataset(train, 0, 1, tok, max_len, True, False)
    data_test = BERTDataset(test, 0, 1, tok, max_len, True, False)

    model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
    model.load_state_dict(torch.load('model/baseline.hdf5'))

    train_model(model, data_train, data_test, device, 20)





if __name__ == "__main__":
    main()