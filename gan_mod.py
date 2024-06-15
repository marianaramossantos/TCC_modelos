import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Configurações
data_dir = './data'
num_epochs = 300
batch_size = 128
lr_g = 0.0002  
lr_d = 0.0001  
latent_dim = 128
label_smoothing = 0.2  

# Verificação de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:"+ {device})

# Transformações de dados
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
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

# Definição do Gerador
class Generator(nn.Module):
    def _init_(self):
        super(Generator, self)._init_()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 3*32*32),  
            nn.Tanh()
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(-1, 3, 32, 32)  
        return img

# Definição do Discriminador
class Discriminator(nn.Module):
    def _init_(self):
        super(Discriminator, self)._init_()
        self.model = nn.Sequential(
            nn.Linear(32*32*3, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),  
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        return self.model(x)

# Inicialização do gerador e do discriminador com pesos normalizados
def weights_init(m):
    classname = m._class.name_
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Inicialização das redes
generator = Generator().to(device)
generator.apply(weights_init)
discriminator = Discriminator().to(device)
discriminator.apply(weights_init)

# Função de perda e otimizadores
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

# Função de regularização 
def add_l2_regularization(model, weight_decay=0.001):
    for param in model.parameters():
        param.data -= param.data * weight_decay

# Função de treinamento 
def train_gan(generator, discriminator, train_loader, criterion, optimizer_G, optimizer_D, num_epochs):
    best_d_accuracy = 0.0
    best_generator = None
    best_discriminator = None

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        for i, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            # Labels para suavização
            valid = (1.0 - label_smoothing) * torch.ones((imgs.size(0), 1), device=device, requires_grad=False)
            fake = label_smoothing * torch.zeros((imgs.size(0), 1), device=device, requires_grad=False)
            
            # Imagens reais
            real_imgs = imgs

            # Treinando o gerador
            optimizer_G.zero_grad()
            z = torch.randn((imgs.size(0), latent_dim), device=device)
            gen_imgs = generator(z)
            g_loss = criterion(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # Treinando o discriminador
            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(real_imgs), valid)
            fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Aplicando regularização 
            add_l2_regularization(generator)
            add_l2_regularization(discriminator)

        print("Epoch [{}/{}], Perda D: {:.4f}, Perda G: {:.4f}".format((epoch+1), num_epochs,d_loss.item(),g_loss.item() ))

        # Avaliação das métricas do discriminador e do gerador
        acc_d, ppv_d, trp_d, fs_d, acc_g, ppv_g, trp_g, fs_g = evaluate_metrics(generator, discriminator, test_loader, val_loader)
        print("Discriminator: ACC: {:.4f}, PPV: {:.4f}, TRP: {:.4f}, FS: {:.4f}".format(acc_d, ppv_d,trp_d,fs_d))
        print("Generator: ACC: {:.4f}, PPV: {:.4f}, TRP: {:.4f}, FS: {:.4f}".format(acc_g, ppv_g, trp_g, fs_g))

        # Salvando o melhor gerador e discriminador com base na acurácia do discriminador
        if acc_d > best_d_accuracy:
            best_d_accuracy = acc_d
            best_generator = generator.state_dict()
            best_discriminator = discriminator.state_dict()
            torch.save(best_generator, 'best_generator.pth')
            torch.save(best_discriminator, 'best_discriminator.pth')
            print(f'Melhor modelo salvo com acurácia do discriminador: {best_d_accuracy:.4f}')

    print('Treinamento da GAN concluído.')

# Função para avaliar o discriminador e o gerador nos conjuntos de teste e validação
def evaluate_metrics(generator, discriminator, test_loader, val_loader):
    generator.eval()
    discriminator.eval()
    y_true_d = []
    y_pred_d = []
    y_true_g = []
    y_pred_g = []
    with torch.no_grad():
        for loader in [test_loader, val_loader]:
            for imgs, _ in loader:
                imgs = imgs.to(device)
                # Avaliação do discriminador
                predicted_d = (discriminator(imgs) > 0.5).float().cpu().numpy().flatten()
                if np.sum(predicted_d) == 0:  # Verificar se não há amostras positivas verdadeiras
                    acc_d = 0.0
                    ppv_d = 0.0
                    trp_d = 0.0
                    fs_d = 0.0
                else:
                    y_true_d.extend(np.ones(imgs.size(0)))  # Imagens reais são consideradas true para calcular a ACC do discriminador
                    y_pred_d.extend(predicted_d)
                # Avaliação do gerador
                z = torch.randn((imgs.size(0), latent_dim), device=device)
                gen_imgs = generator(z)
                predicted_g = (discriminator(gen_imgs) <= 0.5).float().cpu().numpy().flatten()  # As imagens geradas são consideradas fake para calcular a ACC do gerador
                y_true_g.extend(np.zeros(imgs.size(0)))
                y_pred_g.extend(predicted_g)
    
    if len(y_true_d) > 0:
        acc_d = accuracy_score(y_true_d, y_pred_d)
        ppv_d = precision_score(y_true_d, y_pred_d)
        trp_d = recall_score(y_true_d, y_pred_d)
        fs_d = f1_score(y_true_d, y_pred_d)
    else:
        acc_d = 0.0
        ppv_d = 0.0
        trp_d = 0.0
        fs_d = 0.0
    
    acc_g = accuracy_score(y_true_g, y_pred_g)
    ppv_g = precision_score(y_true_g, y_pred_g)
    trp_g = recall_score(y_true_g, y_pred_g)
    fs_g = f1_score(y_true_g, y_pred_g)
    
    return acc_d, ppv_d, trp_d, fs_d, acc_g, ppv_g, trp_g, fs_g

# Treinamento
train_gan(generator, discriminator, train_loader, criterion, optimizer_G, optimizer_D, num_epochs)

# Gerar algumas imagens para avaliação
def generate_images(generator, num_images=25):
    generator.eval()
    z = torch.randn((num_images, latent_dim), device=device)  
    gen_imgs = generator(z)
    return gen_imgs

generated_imgs = generate_images(generator)

# Salvar o gerador treinado
torch.save(generator.state_dict(), 'model_gan.pth')
print('Gerador salvo em model_gan.pth')