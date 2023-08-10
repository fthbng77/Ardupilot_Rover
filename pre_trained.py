import torch
import torch.nn as nn
import torch.optim as optim
from model import SegNet, DoubleConv
from dataset import SegNetDataset
import torchvision.models as models
import matplotlib.pyplot as plt

# Hiperparametreler
batch_size = 32
epochs = 10
lr = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PIN_MEMORY = True if device == 'cuda' else False
TRAIN_IMG_DIR = "Dataset/train/images"
TRAIN_MASK_DIR = "Dataset/train/masks"
VAL_IMG_DIR = "Dataset/test/images"
VAL_MASK_DIR = "Dataset/test/masks"

# VGG16'dan SegNet'e ağırlıkları başlatma fonksiyonu
def initialize_segnet_from_vgg16(segnet_model):
    vgg16 = models.vgg16(pretrained=True)

    # SegNet'in encoder katmanları
    encoders = [segnet_model.encoder1, segnet_model.encoder2, segnet_model.encoder3, segnet_model.encoder4, segnet_model.encoder5]
    
    vgg_features = list(vgg16.features.children())
    vgg_index = 0

    for encoder in encoders:
        for layer in encoder:
            if isinstance(layer, DoubleConv):
                for sub_layer in layer.double_conv:
                    if isinstance(sub_layer, nn.Conv2d) and isinstance(vgg_features[vgg_index], nn.Conv2d):
                        sub_layer.weight.data = vgg_features[vgg_index].weight.data
                        sub_layer.bias.data = vgg_features[vgg_index].bias.data
                        vgg_index += 1
                    elif isinstance(vgg_features[vgg_index], (nn.BatchNorm2d, nn.ReLU)):
                        vgg_index += 1

# Eğitim ve doğrulama veri kümelerini yükleyin
train_dataset = SegNetDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, (224, 224), augment=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=PIN_MEMORY)

val_dataset = SegNetDataset(VAL_IMG_DIR, VAL_MASK_DIR, (224, 224), augment=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=PIN_MEMORY)

model = SegNet(input_channels=3, num_classes=6)
initialize_segnet_from_vgg16(model)
model.to(device)

print("next(model.parameters()).device: ",next(model.parameters()).device)



# Kayıp fonksiyonu ve optimize edici
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Erken durdurma için ayarlar
patience = 3
best_loss = float('inf')
early_stopping_counter = 0

train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).long()

        target = target.squeeze(1)
        
        # Debug: Tip kontrolü
        print("After squeeze, Target dtype:", target.dtype)

        optimizer.zero_grad()
        output = model(data)
        
        # Debug: Tip ve şekil kontrolü
        print("Before Loss, Target dtype:", target.dtype)
        print("Target shape:", target.shape)
        print("Output shape:", output.shape)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Training Average Loss: {avg_train_loss:.4f}")

    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device).long()
            target = target.squeeze(1) 
            print("Target dtype:", target.dtype)
            output = model(data)
            
            # Debug: Tip ve şekil kontrolü
            print("Before Loss in Validation, Target dtype:", target.dtype)
            print("Target shape:", target.shape)
            print("Output shape:", output.shape)
            
            loss = criterion(output, target)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}")


    scheduler.step()

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), 'segnet_best_model.pth')
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print("Erken durdurma gerçekleştirildi!")
            break

print("Eğitim tamamlandı!")

plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss', color='red')
plt.title('Epoch vs. Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_validation_loss.png')
plt.show()
