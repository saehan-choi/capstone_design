import torch
import matplotlib.pyplot as plt

def save_model(model):
    PATH='./weights/'
    return torch.save(model.state_dict(), PATH+'model.pt')

def make_graph(train_loss_arr, val_loss_arr, train_acc_arr, val_acc_arr, epoch_arr):
    plt.plot(epoch_arr, train_acc_arr, 'r', epoch_arr, val_acc_arr, 'b')
    plt.legend(['train_acc','val_acc'])
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()

    plt.plot(epoch_arr, train_loss_arr, 'r', epoch_arr, val_loss_arr, 'b')
    plt.legend(['train_loss','val_loss'])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

