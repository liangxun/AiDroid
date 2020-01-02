import torch
from torch import optim
from torch import nn
from sklearn import metrics
from utils import load_data_for_optim
from model_zoo import GoogLeNet
from config import *


SEED = 2020
torch.manual_seed(SEED)
is_cuda = torch.cuda.is_available()
if is_cuda:
    torch.cuda.manual_seed(SEED)
    torch.cuda.set_device(3)

batchSize = 256
epoch = 5
train_data_loader, test_data_loader = load_data_for_optim(data_path, batchSize)
model = GoogLeNet()
if is_cuda:
    model.cuda()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# =======================train model=========================
hist_loss = []
acc = []
iter = 0
for ep in range(epoch):
    model.train()
    for img, label in train_data_loader:
        if is_cuda:
            img = img.cuda()
            label = label.cuda()
        out = model(img)
        loss = loss_func(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        hist_loss.append(loss.item())
        print('epoch{} iter{}: loss={:.5f}, acc={:.5f}'.format(ep, iter, loss.item(), 
                                                        metrics.accuracy_score(label.cpu().data.numpy(), out.cpu().data.numpy().argmax(axis=1))))
        iter += 1

# ======================test model============================
model.eval()
pred_y = []
true_y = []
for img, label in test_data_loader:
    if is_cuda:
        img = img.cuda()
        label = label.cuda()
    out = model(img)
    pred_y += list(out.cpu().data.numpy().argmax(axis=1))
    true_y += list(label.cpu().data.numpy())

print('f1={:.5f}\tacc={:.5f}\tprecission={:.5f}\trecall={:.5f}'.format(
        metrics.accuracy_score(true_y, pred_y),
        metrics.f1_score(true_y, pred_y),
        metrics.precision_score(true_y, pred_y),
        metrics.recall_score(true_y, pred_y)))
