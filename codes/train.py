import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_cosine_schedule_with_warmup
from utils import calc_accuracy
import copy
from tqdm import tqdm

def train_model(model, train_dataset, test_dataset, device, num_epochs=20):

    # DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=5)
    test_dataloader = DataLoader(test_dataset, batch_size=64, num_workers=5)

    # Optimizer, Loss
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader)*0.1, num_training_steps=num_epochs*len(train_dataloader))

    best_test_acc = 0.0
    best_model = None

    #Training
    for epoch in range(num_epochs):
        model.train()
        train_acc, train_loss = 0.0, 0.0
        for token_ids, valid_length, segment_ids, label in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            optimizer.zero_grad()
            token_ids, segment_ids, label = token_ids.long().to(device), segment_ids.long().to(device), label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            scheduler.step()
            train_acc += calc_accuracy(out, label)
            train_loss += loss.item()

        avg_train_acc = train_acc / len(train_dataloader)
        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Train Accuracy: {avg_train_acc:.4f}, Train Loss: {avg_train_loss:.4f}")

        #Valid
        model.eval()
        test_acc, test_loss = 0.0, 0.0
        with torch.no_grad():
            for token_ids, valid_length, segment_ids, label in tqdm(test_dataloader, desc='Testing'):
                token_ids, segment_ids, label = token_ids.long().to(device), segment_ids.long().to(device), label.long().to(device)
                out = model(token_ids, valid_length, segment_ids)
                loss = loss_fn(out, label)
                test_acc += calc_accuracy(out, label)
                test_loss += loss.item()

        avg_test_acc = test_acc / len(test_dataloader)
        avg_test_loss = test_loss / len(test_dataloader)
        print(f"Epoch {epoch+1} Test Accuracy: {avg_test_acc:.4f}, Test Loss: {avg_test_loss:.4f}")

'''        if avg_test_acc > best_test_acc:
            best_test_acc = avg_test_acc
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, model_path.format(ver=epoch+1))'''
