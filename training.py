import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class ChessDataset(Dataset):
    def __init__(self, rootDir, transform=None):
        self.rootDir = rootDir
        self.transform = transform
        self.classes = ['white', 'black', 'empty']
        self.classToIdx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        
        for className in self.classes:
            classDir = os.path.join(rootDir, className)
            for imgName in os.listdir(classDir):
                if imgName.endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append((os.path.join(classDir, imgName), self.classToIdx[className]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        imgPath, label = self.images[idx]
        image = Image.open(imgPath).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def trainModel(trainLoader, valLoader, numEpochs=30, learningRate=0.001, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ChessNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate)

    for epoch in range(numEpochs):
        model.train()
        runningLoss = 0.0
        
        for inputs, labels in trainLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            runningLoss += loss.item()
            
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in valLoader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Epoch {epoch+1}/{numEpochs}')
        print(f'Training Loss: {runningLoss/len(trainLoader):.4f}')
        print(f'Validation Accuracy: {100 * correct / total:.2f}%')
        print('-' * 40)
    
    return model

def evaluateModel(model, testLoader, device):
    model.eval()
    correct = 0
    total = 0
    classCorrect = [0] * 3
    classTotal = [0] * 3
    confusionMatrix = torch.zeros(3, 3)
    
    with torch.no_grad():
        for inputs, labels in testLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for i in range(len(labels)):
                label = labels[i]
                pred = predicted[i]
                classCorrect[label] += (pred == label).item()
                classTotal[label] += 1
                confusionMatrix[label][pred] += 1

    print(f'\nTest Results:')
    print(f'Overall Accuracy: {100 * correct / total:.2f}%')
    classes = ['white', 'black', 'empty']
    for i in range(3):
        print(f'Accuracy of {classes[i]}: {100 * classCorrect[i] / classTotal[i]:.2f}%')
    
    print('\nConfusion Matrix:')
    print('Predicted →  White   Black   Empty')
    for i in range(3):
        print(f'{classes[i]:<8} {confusionMatrix[i][0]:>7.0f} {confusionMatrix[i][1]:>7.0f} {confusionMatrix[i][2]:>7.0f}')

def testSingleImage(model, imagePath, device, transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    classes = ['white', 'black', 'empty']
    model.eval()
    
    image = Image.open(imagePath).convert('RGB')
    imageTensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(imageTensor)
        _, predicted = torch.max(output, 1)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        print(f'\nImage: {imagePath}')
        print(f'Prediction: {classes[predicted.item()]}')
        print('Probabilities:')
        for i, prob in enumerate(probabilities[0]):
            print(f'{classes[i]}: {prob.item()*100:.2f}%')

def testDirectory(model, testDir, device):
    transform = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    correct = 0
    total = 0
    
    for className in ['white', 'black', 'empty']:
        classDir = os.path.join(testDir, className)
        if not os.path.exists(classDir):
            continue
            
        for imgName in os.listdir(classDir):
            if imgName.endswith(('.png', '.jpg', '.jpeg')):
                imgPath = os.path.join(classDir, imgName)
                trueLabel = className
                
                image = Image.open(imgPath).convert('RGB')
                imageTensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(imageTensor)
                    _, predicted = torch.max(output, 1)
                    predictedClass = ['white', 'black', 'empty'][predicted.item()]
                    
                    isCorrect = predictedClass == trueLabel
                    correct += isCorrect
                    total += 1
                    
                    print(f'\nImage: {imgPath}')
                    print(f'True label: {trueLabel}')
                    print(f'Predicted: {predictedClass}')
                    print(f'Correct: {"✓" if isCorrect else "✗"}')
    
    print(f'\nOverall accuracy: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    dataDir = "squares"
    batchSize = 32
    numEpochs = 30
    learningRate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ChessDataset(dataDir, transform=transform)
    totalSize = len(dataset)
    trainSize = int(0.7 * totalSize)
    valSize = int(0.15 * totalSize)
    testSize = totalSize - trainSize - valSize
    
    trainDataset, valDataset, testDataset = torch.utils.data.random_split(
        dataset, [trainSize, valSize, testSize]
    )
    
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False)
    testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)
    
    model = trainModel(trainLoader, valLoader, numEpochs=numEpochs, learningRate=learningRate, device=device)
    
    testAccuracy = evaluateModel(model, testLoader, device)
    torch.save(model.state_dict(), 'chess_classifier.pth')
    
    testDir = "squaresTest"
    testDirectory(model, testDir, device)