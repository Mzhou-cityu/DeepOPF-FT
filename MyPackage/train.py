import MyPackage.config as config
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from MyPackage.Net import Net
from MyPackage.evaluation_functions import get_clamp
from MyPackage.get_gradient import *
import time

def train():

    model, optimizer, scheduler = train_init()
    print('*' * 8 + 'model training' + '*' * 8)
    # Training process: Voltage angle
    model = train_model(model, optimizer, scheduler)
    save_model(model)

def train_init():
    model = Net(config.input_channels, config.output_channels, config.hidden_units, config.khidden)
    # ************ Load existing models to re-train or randomly initialize the model  ***********
    if (config.pretrain_flag == 1):
        model.load_state_dict(torch.load(config.PATH, map_location=config.device))  # load pre-trained model

    # ********************* set parameters for DNN training ******************************
    optimizer = torch.optim.Adam(model.parameters(), lr=config.Lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=config.gamma)

    if torch.cuda.is_available():
        model.to(config.device)
    return model, optimizer, scheduler

def train_model(model, optimizer, scheduler):
    Fig_loss = []  # save training losses of Va
    Fig_penalty = []
    Fig_mse = []
    criterion = nn.MSELoss()  # loss function
    if (config.pretrain_flag == 1):
        Epoch = config.Epoch_pre
    else:
        Epoch = config.Epoch
    for epoch in range (Epoch):
        running_loss = 0.0
        running_penalty = 0.0
        running_mse = 0.0
        for step, (train_x, train_y) in enumerate(config.training_loader):
            # feedforward
            train_x, train_y = train_x.to(config.device), train_y.to(config.device)
            output_y = model(train_x)
            # if epoch less than specified number/no penalty of V: only MSEloss
            mse = criterion(train_y, output_y)
            loss = config.mse_weight * mse   # defaut loss function
         #   loss = config.mse_weight*mse    # debug the mse loss part
            # backproprogate for Vm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss = running_loss + loss.item()
            running_mse = running_mse + mse.item()
        if (epoch + 1) % config.p_epoch == 0:
            print('epoch  ', epoch + 1, ' loss:  ', round(running_loss, 2))
            print('mse:       ', round(mse.item(), 2))
        #    print(output_y)
        Fig_loss.append(running_loss)
        Fig_mse.append(running_mse)

    return model

def save_model(model):
    #********* Save trained model *********************
    Save_path = config.PATH
    if (config.pretrain_flag == 1):
        Save_path = config.PATH_pre
    torch.save(model.state_dict(), Save_path, _use_new_zipfile_serialization=False)
