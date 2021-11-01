import torch
import torch.nn
from resnet import *
from vggnet import *
from utils import *
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

device = torch.device('cuda')
model = VGG_net(in_channels=3, num_classes=2)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
# parameters 말고 parameters()임.
criterion = nn.CrossEntropyLoss().cuda()



train_data = ImageFolder("./train", 
                            transform=transforms.Compose
                            ([transforms.Resize(226),
                            transforms.RandomCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])
                        )
train_loader = DataLoader(train_data, batch_size=25, shuffle=True)

val_data = ImageFolder("./validation", 
                            transform=transforms.Compose
                            ([transforms.Resize(226),
                            transforms.RandomCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])
                        )
val_loader = DataLoader(val_data, batch_size=25, shuffle=False)

train_loss_arr = []
train_acc_arr = []
val_loss_arr = []
val_acc_arr = []
epoch_arr = []

for epoch in range(1,16):
    train_loss = 0.0
    train_correct = 0

    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        # zero_grad를 안해줘서 정확도가 처음에 안나왔음 ㅠㅠㅠㅠㅠ
        # 지금은 잘 나옴!
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += (predicted == labels).cpu().sum()        
        
    with torch.no_grad():
        val_loss = 0.0
        val_correct = 0
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            outputs = model(inputs.to("cuda"))
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_correct += (predicted == labels).cpu().sum()

    epoch_arr.append(epoch)
    # len을 한 이유는 train값과 val값의 차이가 나서 
    # 그래프를 그리면 정확한비교가 안되기에 평균값으로 바꾸어준다.
    train_loss_arr.append(train_loss/len(train_data)*100)
    train_acc_arr.append(train_correct/len(train_data)*100)
    val_loss_arr.append(val_loss/len(val_data)*100)
    val_acc_arr.append(val_correct/len(val_data)*100)

    print(f'train_loss : {train_loss_arr[-1]}')
    print(f'train_acc : {int(train_acc_arr[-1])}%')

    print(f'val_loss : {val_loss_arr[-1]}')
    print(f'val_acc : {int(val_acc_arr[-1])}%')

    print(f'epoch step :{epoch}')

    
print(f'train_acc_arr is : {train_acc_arr}')


save_model(model)
make_graph(train_loss_arr, val_loss_arr, train_acc_arr, val_acc_arr, epoch_arr)