import torch.nn.functional as F
import torch.optim as optim
import torch, torchvision, numpy
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import Resize, ToTensor
from torchvision.models import vgg16
import matplotlib.pylab as plt
import json
import random
from torchvision.datasets.utils import download_url
import torch.utils.data as Data
import inspect
import sys, os, gc

def define_net():
  net = nn.Sequential(
    nn.Conv2d(6, 20, kernel_size = 5, stride=1),
    nn.BatchNorm2d(20),
    nn.MaxPool2d(4, stride = 4),
    nn.Conv2d(20, 50, kernel_size = 5, stride = 1),
    nn.BatchNorm2d(50),
    nn.MaxPool2d(3, stride = 3),
    nn.ELU(),
    nn.Conv2d(50, 3, kernel_size = 1, stride = 1)
  )
  return net

def train(net, data, classes, test_features, test_classes):
  criterion = nn.CrossEntropyLoss()
  # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
  optimizer = optim.Adam(net.parameters(), lr=0.05)

  t_dataset = Data.TensorDataset(data, classes)

  samples_weight = [1/5, 1/5, 1/5]

  sampler = Data.WeightedRandomSampler(samples_weight, 30)

  # Batch size is one
  loader = Data.DataLoader(dataset=t_dataset, batch_size=1, num_workers=0, shuffle=True)

  d = data[0]
  c = classes[0]
  co_far = 0

  print()
  print()
  print()

  for epoch in range(99):  # loop over the dataset multiple times
    ct = 0
    running_loss = 0.0
    for i, data in enumerate(loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        outputs = torch.reshape(outputs, (1,3))
        # print(outputs.tolist())

        # print(outputs)
        # print(labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics

        running_loss += loss.item()
        ct += 1
    if ((epoch+1) % 10 == 0):

      record_acc(net, test_features, test_classes)

      # cor = torch.argmax(net(torch.Tensor([d]))).item() == c.item()
      # if cor:
      #   co_far += 1
      # print(cor)
      # print()
      print('[%d] loss: %.8f' % (epoch + 1,  running_loss/ct))
      ct = 0
      running_loss = 0.0

  return net

def get_images():
    transform = transforms.Compose( [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    features = []
    classes = []

    with open("filenames.txt") as fh:

      for line in fh:
        parts = line.rstrip().split('_')
        c = int(parts[1])
        clas = c
        if c == 0:
          clas = 1
        elif c == 2 or c == 1:
          clas = 0
        elif c == 3:
          clas = 2
        classes.append(clas)

        features.append(torch.load('resized_tensors/{}.pt'.format(line.rstrip())))
        # features.append([torch.load('resized_tensors/{}.pt'.format(line.rstrip()))])

    return classes, features

def train_net():

  target, inputs = get_images()

  groups, class_groups = cross_fold_n(inputs, target, 4)


  for i in range(len(groups)):
    test_features = groups[i]
    test_classes = class_groups[i]

    train_features = [item for sublist in (groups[:i] + groups[i+1:]) for item in sublist]
    train_classes = [item for sublist in (class_groups[:i] + class_groups[i+1:]) for item in sublist]


    target = torch.tensor(train_classes, dtype=torch.long)
    inputs = torch.stack(train_features)


    net = define_net()
    net = train(net, inputs, target, test_features, test_classes)

    record_acc(net, test_features, test_classes)


def class_count(X, Y, fraction):
  counts = {}
  sep_data = {}

  for k in range(4):
    counts[k] = 0
    sep_data[k] = []

  for x, y in zip(X, Y):
    counts[y] = counts[y] + 1
    sep_data[y] = sep_data[y] + [x]

  train = []
  test = []
  train_classes = []
  test_classes = []


  for k in sep_data:
    train += sep_data[k][:int(len(sep_data[k])*fraction)]
    test += sep_data[k][int(len(sep_data[k])*fraction):]
    train_classes += [k] * int(len(sep_data[k])*fraction)
    test_classes += [k] * (int(len(sep_data[k])) - int(len(sep_data[k])*fraction))

  return train, test, train_classes, test_classes

def cross_fold_n(X, Y, n):
  counts = {}
  sep_data = {}

  for k in range(3):
    counts[k] = 0
    sep_data[k] = []

  for x, y in zip(X, Y):
    counts[y] = counts[y] + 1
    sep_data[y] = sep_data[y] + [x]

  groups = [[]] * n
  class_groups = [[]] * n

  max_len = 0
  for k in sep_data:
    max_len = (len(sep_data[k]) if len(sep_data[k]) > max_len else max_len)

  for k in sep_data:
    sep_data[k] = sep_data[k] * (max_len//len(sep_data[k]))

  for k in sep_data:
    random.shuffle(sep_data[k])
    for i in range(n):
      begin = i*len(sep_data[k])//n
      end = min(len(sep_data[k]), (i + 1)*len(sep_data[k])//n)
      length = end - begin

      groups[i] = groups[i] + sep_data[k][begin:end]
      class_groups[i] = class_groups[i] + [k] * length

  return groups, class_groups

def record_acc(net, testx, testy):

  t_dataset = Data.TensorDataset(torch.stack(testx), torch.tensor(testy, dtype=torch.long))

  # Batch size is one
  loader = Data.DataLoader(dataset=t_dataset, batch_size=1, shuffle=True, num_workers=0)


  counts = [0, 0, 0]
  correct = [0, 0, 0]
  total = [0, 0, 0]

  for i, data in enumerate(loader, 0):
    inputs, labels = data
    pred = net(inputs)
    help = torch.tensor([[0, 0, 1]], dtype=torch.long)
    # print(help)
    # print(torch.argmax(help))
    for i in range(len(labels)):
      # print(torch.reshape(pred[i], (1,3)))
      p = torch.argmax(torch.reshape(pred[i], (1,3)))
      l = labels[i]
      counts[p.item()] += 1
      if (p.item() == l.item()):
        correct[l.item()] += 1

      total[l.item()] += 1

  for i in range(len(correct)):
    c = correct[i]/total[i]
    print('Acc. class: {}'.format(i), c)
  print("Acc. Overall: {}".format(sum(correct)/sum(total)))
  print("Each class predictions: {}".format(str(counts)))
  print("Each class totals     : {}".format(str(total)))

def make_batch(filenames):
  ims = []
  for file in filenames:
    image = Image.open(file).convert('RGB')
    tensor_image = image_transform(image)
    ims.append(tensor_image)
  batch = torch.stack([ti[:, -320:, :] for ti in ims]).to("cuda")
  return (batch, filenames)

def make_batches(files, n=10):
  chunks = [files[i:i + n] for i in range(0, len(files), n)]
  return [make_batch(x) for x in chunks]


def test_net(filename):
  a = filename + "_rgb.jpg"
  b = filename + "_irdemortho.jpg"

  batches = [make_batch([a, b])]
  with open("train_net" + ".features", 'a') as ffile:
    for batch, bfiles in batches:

      pred = Predictor().to("cuda")
      torch.cuda.empty_cache()
      res, pred_vec = pred(batch)

      for i, p in enumerate(pred.outputs[0]):
        pre, ext = os.path.splitext(bfiles[i])
        id, c, x, y, type = pre.split("/")[-1].split("_")

        ffile.write("{}|{}|{}|{}|{}\n".format(id, c, x, y, str(p.detach().cpu().numpy().tolist())))
        # numpy.savetxt(pre + ".features", p.detach().cpu().numpy())

      del res
      del pred_vec
      del pred
      gc.collect()


train_net()
