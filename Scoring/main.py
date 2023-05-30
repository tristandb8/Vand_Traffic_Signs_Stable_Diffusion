import torchvision
import glob
from PIL import Image
import torchvision.transforms as transforms
import torch
from dl import data_loader
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import time
import os
import sys



def main(argv):
    batch_size = 32
    vandTrain = True
    save_model = 'weights/pt_VT_final1'
    epochs = 20

    print("batch_size:", batch_size, ", vandTrain:", vandTrain, ", weights path:", save_model, ", num epochs:", epochs)

    dl = data_loader(vandTrain = vandTrain)
    train_dataloader = DataLoader(dl, batch_size=batch_size, shuffle=True, num_workers=0, drop_last = True)
    print("Size of dl", len(train_dataloader), flush=True)

    model = torchvision.models.swin_v2_s(weights = "Swin_V2_S_Weights.IMAGENET1K_V1").cuda()#weights = "Swin_V2_S_Weights.IMAGENET1K_V1"
    # modify the classifier head
    in_features = model.head.in_features
    model.head = torch.nn.Linear(in_features, 2).cuda()
    print("model loaded")

    if torch.cuda.device_count()>1:
        print(f'Multiple GPUS found!')
        model=torch.nn.DataParallel(model)
        model.cuda()
    else:
        print('Only 1 GPU is available')
        model.cuda()

    if not os.path.exists(save_model):
        os.makedirs(save_model)
    else:
        extension = 1
        while True:
            new_save_model = save_model + '_ext{}'.format(extension)
            if not os.path.exists(new_save_model):
                os.makedirs(new_save_model)
                break
            extension += 1
        save_model = new_save_model

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.05)
    criterion= torch.nn.CrossEntropyLoss().cuda()

    for epoch in range(epochs):
        losses = []
        model.train()
        for i, data in enumerate(train_dataloader, 0):
            image, label = data
            #print(image.shape)
            if (i == 0) & (epoch == 0):
                print("epoch 0 started", image.shape, flush=True)
            image = Variable(image.cuda())
            label = torch.as_tensor(label)
            label = Variable(label.cuda())
            optimizer.zero_grad()
            pred = model(image)
            loss = criterion(pred,label.long())
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if ((i % int(13623 / batch_size / 10) == 0) & (epoch == 0)):
                print(i / int(13623 / batch_size), "% progress", flush=True)
        print("End of epoch " + str(epoch) + ", mean loss: " + str(np.mean(losses)), flush=True)
        
        if (epoch % 5 == 0):
            validate(model, dl)
            print(save_model+"/epoch"+str(epoch).zfill(6)+'.pt')
            torch.save(model.state_dict(), save_model+"/epoch"+str(epoch).zfill(6)+'.pt')


def validate(model, dl):
    dl_val = data_loader(val = True)
    val_dataloader = DataLoader(dl_val, batch_size=1, shuffle=True, num_workers=0, drop_last = True)
    model.eval()
    correct = {}
    categories = ["ImageNet", "GTSRB", "Kaggle", "TSRD", "vandSigns"]
    for cat in categories:
        correct[cat] = []

    for i, data in enumerate(val_dataloader, 0):
        image, label, cat = data
        cat = cat[0]
        image = Variable(image.cuda())
        label = torch.as_tensor(label)
        label = Variable(label.cuda())
        pred = model(image)[0]
        sftmx = torch.nn.Softmax(dim=0)
        pred = sftmx(pred)
        pred = torch.tensor([torch.argmax(pred)]).cuda()
        correct[cat].append(int(pred == label))
    for cat in categories:
        print("final test acc", cat, ":", np.mean(correct[cat]))
    model.train()


if __name__ == "__main__":
   main(sys.argv[1:])