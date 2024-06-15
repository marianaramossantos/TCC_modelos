import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Configurações
data_dir = './data'
num_epochs = 50  
batch_size = 128 
learning_rate = 0.001  # Taxa de aprendizado inicial

# Transformações com aumento de dados
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalização padrão do CIFAR-100
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalização padrão do CIFAR-100
])

# Carregamento do conjunto de dados
train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)

# Divisão do conjunto de dados de treinamento em treinamento e validação
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Definindo a LSTM
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)  # Adicionando dropout
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Redimensionar x para ser 3D: (batch_size, sequence_length, input_size)
        x = x.view(x.size(0), -1, x.size(1))  # ajustar para o formato correto
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        
        out = self.fc(out[:, -1, :])
        return out

# Dimensões de entrada e de saída da LSTM
input_size = 3  # Número de canais de entrada RGB
hidden_size = 128  # Aumentando o tamanho do vetor de estado oculto
num_layers = 2  # Ajustando para 2 camadas LSTM
num_classes = 100

# Inicializar o modelo LSTM
model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes)

# Função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Função de treinamento
def train_rnn(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    print("Iniciando treinamento...")
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
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print("Epoch {}/{}, Perda: {:.4f}".format((epoch+1), num_epochs, epoch_loss))
    
    print("Treinamento concluído.")

# Função de avaliação
def evaluate_rnn(model, test_loader):
    print("Avaliando o modelo...")
    model.eval()
    corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            corrects += torch.sum(preds == labels)
    
    test_acc = corrects.double() / len(test_loader.dataset)
    print("Acurácia no teste: {:.4f}".format(test_acc))
    if test_acc > 0.7:
        print("Os resultados são bons!")
    else:
        print("Os resultados podem ser melhorados.")

# Treinar o modelo LSTM
train_rnn(model, train_loader, val_loader, criterion, optimizer, num_epochs)

# Avaliar o modelo LSTM
evaluate_rnn(model, test_loader)

# Salvar o modelo treinado
torch.save(model.state_dict(), 'model_lstm.pth')
print('Modelo salvo em model_lstm.pth')