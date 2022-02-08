from sklearn.utils import shuffle
from wandb import wandb
import dataset
import torch
from torch.utils import data
import pickle
import model
from torch import nn, optim
import dataset

spectra_size = 50000

def train_classifier(config=None):
    with wandb.init(config=config):
        config = wandb.config
        wandb.name = "Sweep "+str(config)
        net = model.Net(spectra_size,config.layer1_size)
        print("this train config: ",config)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                net = nn.DataParallel(net)
        net.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = build_optimizer(net, config.optimizer,config.learning_rate)

        train_dataset = dataset.SpectraDataset("../pickle_files/train_specs.pkl",spectra_size,25000)
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=25,shuffle=True,num_workers=4)
        print("train_dataset length: ",len(train_dataset))
        print("train_loader length: ",len(train_loader))

        val_dataset = dataset.SpectraDataset("../pickle_files/val_specs.pkl",spectra_size,5000)
        val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=25,shuffle=True,num_workers=4)
        print("val_dataset length: ",len(val_dataset))
        print("val_loader length: ",len(val_loader))

        for epoch in range(config.epochs):
            running_loss = 0.0
            epoch_steps = 0
            print("epoch: ",epoch)
            for i, data in enumerate(train_loader, 0):
                #print("batch: ",i)
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                #print("to devices done")

                # zero the parameter gradients
                optimizer.zero_grad()
                #print("zero grad done")

                # forward + backward + optimize
                outputs = net(inputs)
                #print("forward pass done")
                loss = criterion(outputs, labels)
                #print("loss calculation done")
                loss.backward()
                optimizer.step()
                #print("optimizer step done")

                # print statistics
                running_loss += loss.item()
                epoch_steps += 1

                #print("batch loss: ",loss.item())

                #wandb.log({"batch loss":loss.item()})

            # Validation loss
            val_loss = 0.0
            val_steps = 0
            total = 0
            correct = 0
            for i, data in enumerate(val_loader, 0):
                with torch.no_grad():
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = net(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    loss = criterion(outputs, labels)
                    val_loss += loss.cpu().numpy()
                    val_steps += 1

            wandb.log({"val_loss":(val_loss/val_steps),"epoch":epoch})
            wandb.log({"avg_loss":(running_loss/len(train_loader)),"epoch":epoch})
            wandb.log({"accuracy":(correct/total),"epoch":epoch})
        print("Finished Training")
        
def test_accuracy(net, device="cpu"):
    test_dataset = dataset.SpectraDataset("../pickle_files/test_specs.pkl",spectra_size)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=4,shuffle=True,num_workers=8)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            input, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def build_optimizer(network,optimizer,learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),lr=learning_rate)
    return optimizer


"""
for local_batch,local_labels in train_generator:
    print("local_batch: ",local_batch)
    print("local_labels: ",local_labels)
    break
"""


