import torch
import torch.nn as nn
import Code.SLDLoader.sld_loader as SLDLoader
import numpy as np
import random
import os
import argparse
from tqdm import tqdm

class ModifiedLightweight3DCNN(nn.Module):
    def __init__(self):
        super(ModifiedLightweight3DCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=30, out_channels=8, kernel_size=(3, 3), stride=1, padding=1)
        self.res2 = self._make_layer(block_count=3, in_channels=8, out_channels=8)  # Modified in_channels
        self.res3 = self._make_layer(block_count=4, in_channels=8, out_channels=64)  # Modified in_channels
        self.res4 = self._make_layer(block_count=6, in_channels=64, out_channels=128)
        self.res5 = self._make_layer(block_count=3, in_channels=128, out_channels=256)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1)

    def _make_layer(self, block_count, in_channels, out_channels):
        layers = []
        for _ in range(block_count):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels  # Update in_channels for the next layer
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
    
def init_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(data_folder, highlight_sign_id, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sign_list = sorted(os.listdir(data_folder))
    highlight_sign = sign_list[highlight_sign_id]
    os.makedirs(save_path, exist_ok=True)

    # Check if the model already exists
    exist_ = False
    for file in os.listdir(save_path):
        if highlight_sign in file:
            exist_ = True
            break
    if exist_:
        print(f'{highlight_sign} already trained')
        return

    dataset = SLDLoader.SLD(data_folder, 30, 32, 42)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset.get_generator(highlight_sign, num_data=128),
        batch_size=32,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=lambda _: init_seed(42)
    )
    model = ModifiedLightweight3DCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in tqdm(range(100), desc="Training"):
        for i, (data, label) in enumerate(data_loader):
            data = data.to(device)
            label = label.to(device)
            data = torch.einsum('b t w c -> b t c w', data)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            #print(f'Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}')

    # Test the model
    model.eval()
    correct = 0
    total = 0
    TP = 0
    FP = 0
    FN = 0
    for j in range(10):
        for i, (data, label) in enumerate(data_loader):
            data = data.to(device)
            label = label.to(device)
            data = torch.einsum('b t w c -> b t c w', data)
            output = model(data)
            output = output > 0.5
            total += label.size(0)
            correct += (output == label).sum().item()
            TP += ((output == label) & (output == 1)).sum().item()
            FP += ((output != label) & (output == 1)).sum().item()
            FN += ((output != label) & (output == 0)).sum().item()
    accuracy = correct / total
    precision = TP / ((TP + FP) if TP + FP != 0 else 1)
    recall = TP / ((TP + FN) if TP + FN != 0 else 1)
    f1 = 2 * precision * recall / ((precision + recall) if precision + recall != 0 else 1)
    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')
    
    save_name = os.path.join(save_path, f'{highlight_sign}_{round(accuracy * 100)}_{round(precision * 100)}.pth')
    torch.save(model.state_dict(), save_name)


def process_range(data_folder, save_path, start, end):
    print(f"Training from {start} to {end}")
    for i in range(start, end):
        train(data_folder, i, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', '-i', type=str, default='data')
    parser.add_argument('--save_path', '-o', type=str, default='models')
    parser.add_argument('--start', '-s', type=int, default=0)
    parser.add_argument('--end', '-e', type=int, default=10)
    args = parser.parse_args()
    init_seed(42)
    process_range(args.data_folder, args.save_path, args.start, args.end)
