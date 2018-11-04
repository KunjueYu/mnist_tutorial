import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time
from torch.autograd import Variable

BATCH_SIZE = 128
NUM_EPOCHS = 10

# preprocessing
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# download and load the data
train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform, download=False)



# encapsulate them into dataloader form
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# D_in = 784
# H = 100
# D_out = 10
class SimpleNet(nn.Module):
# TODO:define model
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(SimpleNet, self).__init__()
        self.linear1 = torch.nn.Linear(784, 100)
        self.linear2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred




    
model = SimpleNet()

# TODO:define loss function and optimiter
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)


# train and evaluate
for epoch in range(NUM_EPOCHS):
    for images, labels in tqdm(train_loader):
        # TODO:forward + backward + optimize
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels).unsqueeze(1)
        #print(labels.type)
        labels = torch.zeros(BATCH_SIZE, 10).scatter_(1, labels, 1.0)
        y_pred = model(images)

    # Compute and print loss
        loss = criterion(y_pred, labels)
        #print(y_pred)
        #print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

total = 0
correct = 0

for images, labels in test_loader:

    images = Variable(images.view(-1, 28 * 28))
    outputs = model(images)
    #output = output.long().squeeze()

    _, predicts = torch.max(outputs.data, 1)
    total += labels.size(0)

    correct += (predicts == labels).sum()
print('%d'%total)
print('%d'%correct)
print('Accuracy = %0.2f%%' % (100.0 * float(correct) / float(total)))
total = 0
correct = 0
for images, labels in train_loader:

    images = Variable(images.view(-1, 28 * 28))
    outputs = model(images)
    

    _, predicts = torch.max(outputs.data, 1)
    total += labels.size(0)

    correct += (predicts == labels).sum()
print('%d'%total)
print('%d'%correct)
print('Accuracy = %0.2f%%' % (100.0 * float(correct) / float(total)))

    # TODO:calculate the accuracy using traning and testing dataset

