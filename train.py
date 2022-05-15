import argparse
from utils import get_training_dataloader, get_test_dataloader, WarmUpLR
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from models.densenet import DenseNet
import time
import os
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('-b',
                    type=int,
                    default=128,
                    help='batch size for dataloader')
parser.add_argument('-epoch',
                    type=int,
                    default=100,
                    help='warm up training phase')
parser.add_argument('-test_name',
                    type=str,
                    default='default',
                    help='wandb test name')
parser.add_argument('-lr',
                    type=float,
                    default=0.1,
                    help='initial learning rate')
parser.add_argument('-resume',
                    action='store_true',
                    default=False,
                    help='resume training')
parser.add_argument('-cutout',
                    action='store_true',
                    default=False,
                    help='cutout flag')

parser.add_argument('-adam',
                    action='store_true',
                    default=False,
                    help='adam flag')
args = parser.parse_args()

print('=' * 20)
print(args)
print('=' * 20)

wandb.init(name=args.test_name,
           project="Garbage classification",
           entity="aorus")
wandb.config = args

# Load data
# datas = numpy.load('data/training_20_32.npz.npy', allow_pickle=True)
# datas = numpy.load('data/training_7_224.npz.npy', allow_pickle=True)
# datas = numpy.load('data/training_7_64.npz.npy', allow_pickle=True)
datas = numpy.load('data/training_32.npz.npy', allow_pickle=True)
factor_num = 0.1
numpy.random.seed(123456)
numpy.random.shuffle(datas)
split_idx = -1 * int(len(datas) * factor_num)
train_datas = datas[:split_idx]
test_datas = datas[split_idx:]

train_dataloader = get_training_dataloader(train_datas,
                                           batch_size=args.b,
                                           cutout=args.cutout)
test_dataloader = get_test_dataloader(test_datas)

net = DenseNet(growthRate=12,
               depth=100,
               reduction=0.5,
               bottleneck=True,
               nClasses=4)

if args.resume and os.path.exists(
        'ckpt/bs128_lr3e-05_epoch60_bestacc0.699324369430542_sgd.pt'):
    net.load_state_dict(
        torch.load(
            'ckpt/bs128_lr3e-05_epoch60_bestacc0.699324369430542_sgd.pt',
            map_location='cpu'))

loss_function = nn.CrossEntropyLoss()
if args.adam:
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
else:
    optimizer = optim.SGD(net.parameters(), lr=args.lr)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net.to(device)


def train(epoch):

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(train_dataloader):

        labels = labels.to(device)
        images = images.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        if batch_index != 0 and batch_index % 10 == 0:
            print(
                'Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'
                .format(loss.item(),
                        optimizer.param_groups[0]['lr'],
                        epoch=epoch,
                        trained_samples=batch_index * args.b + len(images),
                        total_samples=len(train_dataloader.dataset)))
            wandb.log({"train loss": loss.item()})
            wandb.log({"lr": optimizer.param_groups[0]['lr']})

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(
        epoch, finish - start))


@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for (images, labels) in test_dataloader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    # if device == 'cuda':
    #     print('GPU INFO.....')
    #     print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print(
        'Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'
        .format(epoch, test_loss / len(test_dataloader.dataset),
                correct.float() / len(test_dataloader.dataset),
                finish - start))
    print()

    wandb.log({"test average loss": test_loss / len(test_dataloader.dataset)})
    wandb.log(
        {"test accuracy": correct.float() / len(test_dataloader.dataset)})

    return correct.float() / len(test_dataloader.dataset)


best_acc = float('-inf')

for epoch_idx in range(args.epoch):

    train(epoch_idx)
    acc = eval_training(epoch_idx)

    if epoch_idx != 0 and epoch_idx % 5 == 0:
        if acc > best_acc:
            best_acc = acc
            torch.save(
                net.state_dict(),
                'ckpt/bs{}_lr{}_epoch{}_bestacc{}_sgd.pt'.format(
                    args.b, args.lr, epoch_idx, best_acc))
