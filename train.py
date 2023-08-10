import torch
import torch.nn as nn
import torch.optim as optim
from model import SegNet
from dataset import SegNetDataset

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

# Eğitim veri kümesini yüklemek
train_dataset = SegNetDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, (224, 224), augment=True)
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    pin_memory=PIN_MEMORY
)

# Doğrulama veri kümesini yüklemek
val_dataset = SegNetDataset(VAL_IMG_DIR, VAL_MASK_DIR, (224, 224), augment=False)
val_loader = torch.utils.data.DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    pin_memory=PIN_MEMORY
)

# Model, kayıp fonksiyonu ve optimize edici
model = SegNet(input_channels=3, num_classes=6).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Erken Durdurma için Ayarlar
patience = 3
best_loss = float('inf')
early_stopping_counter = 0

train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Training Average Loss: {avg_train_loss:.4f}")

    # Doğrulama döngüsü
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}")
    
    scheduler.step()

    # Erken Durdurma Kontrolü
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

import matplotlib.pyplot as plt

plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss', color='red')
plt.title('Epoch vs. Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_validation_loss.png')
plt.show()

