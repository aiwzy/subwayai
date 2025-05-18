import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt

#数据加载
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for label in sorted(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path) and label.isdigit():
                for img_path in glob.glob(os.path.join(label_path, '*.*')):
                    if img_path.lower().endswith(('.png', '.jpg')):
                        self.image_paths.append(img_path)
                        self.labels.append(int(label))

        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path) 
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.labels[idx]

#模型
class CNNLSTMModel(nn.Module):
    def __init__(self, num_classes=5):
        super(CNNLSTMModel, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)

            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten()
        
        )

        self.fc1 = nn.Linear(256, 5)

        

    def forward(self, x):
        features = self.cnn(x)
        output = self.fc(features)
        return output


#训练
def train():
    data_root = './train_data'
    batch_size = 32
    num_epochs = 100
    lr = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    plt.ion()  
    fig = plt.figure()
    ax = fig.add_subplot(111)

    transform = transforms.Compose([
        #transforms.Resize((128, 128)),        
        transforms.ToTensor(),
        # transforms.Normalize(                
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # )
    ])

    dataset = ImageDataset(root_dir=data_root, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CNNLSTMModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=5,         
                    T_mult=2,       
                    eta_min=0.00001    
                )

    epoch_losses = []
    epoch_val_losses = []

    best_val_acc = 0.0
    patience = 5
    no_improve_epochs = 0
    best_model_weights = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (batch_idx+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Acc: {100*correct/total:.2f}%')
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        scheduler.step()
        epoch_losses.append(epoch_loss)

        print(f'\nEpoch [{epoch+1}/{num_epochs}] Complete: '
              f'Avg Loss: {epoch_loss:.4f}, Avg Acc: {epoch_acc:.2f}%\n')
        
        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_total += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()


        epoch_val_loss = val_loss_total / len(val_loader) 
        epoch_val_losses.append(epoch_val_loss) 
        
        val_acc = 100 * val_correct / len(val_dataset)
        print(f'Validation Acc: {val_acc:.2f}%, Val Loss: {epoch_val_loss:.4f}')

        if val_acc > best_val_acc:
            print(f'Validation accuracy improved from {best_val_acc:.2f}% to {val_acc:.2f}%')
            best_val_acc = val_acc
            best_model_weights = model.state_dict().copy() 
            no_improve_epochs = 0
            torch.save(best_model_weights, 'best_model_weights.pth')
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience and epoch > 30:
            print(f'\nEarly stopping triggered at epoch {epoch+1}!')
            model.load_state_dict(best_model_weights) 
            torch.save(best_model_weights, 'best_model_weights.pth') 
            break 

        print(f'Best validation accuracy: {best_val_acc:.2f}%')

        ax.clear()
        ax.plot(range(1, epoch+2), epoch_losses, 'b-', label='Train Loss')
        ax.plot(range(1, epoch+2), epoch_val_losses, 'r-', label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        plt.draw()
        plt.pause(0.1)


    
    plt.ioff()
    plt.show()

    print('训练完成！')

if __name__ == '__main__':
    train()
    