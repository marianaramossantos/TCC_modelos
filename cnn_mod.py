import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

# Configurações
data_dir = './data'
batch_size = 128
learning_rate = 0.001
num_epochs = 100
patience = 5  # Número máximo de epochs sem melhoria na validação

# Transformações com aumento de dados avançado
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Carregamento do conjunto de dados
train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)

# Divisão do conjunto de dados de treinamento em treinamento e validação
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Modelo CNN 
model = models.resnet50(pretrained=True)

# Descongele todas as camadas
for param in model.parameters():
    param.requires_grad = True

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 100)
)

# Função de perda e otimizador 
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# Função de treinamento com parada antecipada
def train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience):
    print("Iniciando treinamento...")
    best_val_acc = 0.0
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        scheduler.step()  # Ajusta a taxa de aprendizado

        epoch_loss = running_loss / len(train_loader.dataset)
        print("Epoch {}/{}, Loss: {:.4f}".format(epoch,(num_epochs - 1),epoch_loss))
        
        # Validação
        model.eval()
        val_loss = 0.0
        corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, 1)
                corrects += torch.sum(preds == labels.data)
            
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = corrects.double() / len(val_loader.dataset)
        print("Validação de perdas: {:.4f}, Validação de Acuracia: {:.4f}".format(val_loss,val_acc))
        
        # Verificação da acurácia de validação 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Verificação se houve melhoria nas últimas épocas
        if epochs_without_improvement >= patience:
            print("Parando o treinamento. Não houve melhoria por {} épocas.".format(patience))
            break
    
    print("Treinamento concluído.")
    return best_val_acc

# Treinar o modelo com parada antecipada
best_val_accuracy = train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience)

# Avaliar o modelo
def evaluate_model(model, test_loader):
    print("Avaliando o modelo...")
    model.eval()
    corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            corrects += torch.sum(preds == labels.data)
    
    test_acc = corrects.double() / len(test_loader.dataset)
    print('Teste de Acuracia: {:.4f}'.format(test_acc))
    if test_acc > 0.7:
        print("Os resultados são bons!")
    else:
        print("Os resultados podem ser melhorados.")

evaluate_model(model, test_loader)

# Salvar o modelo treinado
torch.save(model.state_dict(), 'model_cnn.pth')
print('Model saved to model_cnn.pth')
