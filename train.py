from model import ViP
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.manual_seed(0) # You should have 92.3% acc in epoch 3

model = ViP(
    image_pix = 28,
    patch_pix = 2, # this will result in 14 * 14 words
    class_cnt = 10,
    layer_cnt = 3,
    kernel_ratio = 0.8
)

device = "cuda:0"
batch_size = 128
lr = 2e-5
epochs = 3

transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
])

mnist_train = datasets.MNIST(root="MNIST/",
                             train=True,
                             download=True,
                             transform=transform)

mnist_test = datasets.MNIST(root="MNIST/",
                             train=False,
                             download=True,
                             transform=transform)

data_train = DataLoader(dataset=mnist_train,
                        batch_size=batch_size,
                        shuffle=True)

data_test= DataLoader(dataset=mnist_train,
                        batch_size=batch_size//4,
                        shuffle=False)

opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = 1e-8)
criterion = nn.CrossEntropyLoss()
model.to(device)

for epoch in range(1, epochs + 1):
    acc = 0
    tot_loss = 0
    train_cnt = 0
    test_cnt = 0
    model.train()
    pbar = tqdm(data_train)

    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)  
        opt.zero_grad()
        loss = criterion(y_pred, y)
        loss.backward()
        opt.step()
        tot_loss += loss.item()*x.shape[0]
        train_cnt += x.shape[0]
        pbar.set_description(f"Loss : {tot_loss/train_cnt:.4f}")
        
    model.eval()

    for x, y in data_test:
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        y_argmax = y_pred.argmax(dim = 1)
        acc += (y == y_argmax).sum()
        test_cnt += x.shape[0]
    
    print(f'epoch {epoch} : Average loss : {tot_loss/train_cnt:.4f}, test_acc : {acc.item()/test_cnt:.4f}')

