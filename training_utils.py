import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from utils import * 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json

def get_data_loaders(batch_size, img_size, root_dir = None, normalize = None, already_split = True):
    train_transforms_list = [
        transforms.Resize((img_size, img_size)),  # Resize the images to 224x224
        transforms.RandomRotation(degrees=10),  # Randomly rotate images by up to 10 degrees
        transforms.RandomHorizontalFlip(),  # Randomly flip the images horizontally
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Randomly changes brightness, contrast, and saturation
        transforms.RandomAdjustSharpness(sharpness_factor=2),  # Randomly adjust the sharpness
        transforms.ToTensor()  # Convert the PIL Image to a PyTorch tensor
    ]
    
    val_test_transforms_list = [
        transforms.Resize((img_size, img_size)),  # Resize the images to 224x224
        transforms.ToTensor()  # Convert the PIL Image to a PyTorch tensor
    ]
    

    # Conditionally add the normalization transformation
    if normalize:
        train_transforms_list.append(normalize)
        val_test_transforms_list.append(normalize)

    train_transforms = transforms.Compose(train_transforms_list)
    val_test_transforms = transforms.Compose(val_test_transforms_list)
    
    
    if already_split:
        train_dataset = datasets.ImageFolder(root=root_dir + '/train', transform=train_transforms)
        valid_dataset = datasets.ImageFolder(root=root_dir + '/val', transform=val_test_transforms)
        test_dataset = datasets.ImageFolder(root=root_dir + '/test', transform=val_test_transforms)
    else:
        dataset = datasets.ImageFolder(root=root_dir)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.2 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        train_dataset.dataset.transform = train_transforms
        valid_dataset.dataset.transform = val_test_transforms
        test_dataset.dataset.transform = val_test_transforms
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader

def train_epoch(model, optimizer, criterion, train_loader, device=None):
    device = device or torch.device("cpu")
    model.train()

    train_loss = 0.0
    correct_predictions = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        prob = torch.sigmoid(outputs.data)
        predicted = prob > 0.5
        correct_predictions += (predicted == targets).sum().item()
        
    train_loss /= len(train_loader.dataset)
    train_accuracy = correct_predictions / len(train_loader.dataset)
        
    return train_loss, train_accuracy
        
def test(model, criterion, valid_loader, device = None):
    correct_predictions = 0
    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()
            prob = torch.sigmoid(outputs.data)
            predicted = prob > 0.5
            correct_predictions += (predicted == targets).sum().item()
            
    test_accuracy = correct_predictions / len(valid_loader.dataset)
    test_loss /= len(valid_loader.dataset)
    
    return test_loss, test_accuracy
            

def train(config, 
          root_data_dir,
          num_epochs=10, 
          chkpt_interval=5, 
          checkpoint_dir='./cls_chkpoint', 
          early_stopping_patience=3, 
          resume_checkpoint_path=None, 
          scheduler=None, 
          device = "cuda",
          model_type = "resnet18",
          already_split = True
          ):
    
    now = datetime.datetime.now()
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    if model_type == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)
    train_loader, valid_loader, test_loader = get_data_loaders(config["batch_size"], config["img_size"], root_data_dir, normalize = config["normalize"], already_split = already_split)
    
    if config["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config['momentum'], nesterov=config["nestrov"])
    elif config["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=config["lr"])
    else:
        raise ValueError(f"Invalid optimizer {config['optimizer']}")
    
    checkpoint_dir = os.path.join(checkpoint_dir, now.strftime('checkpoint_%Y-%m-%d-%H-%M') + f'_lr_{optimizer.param_groups[0]["lr"]}_batch_{train_loader.batch_size}')
    now = datetime.datetime.now()
    checkpoint_dir = os.path.join(checkpoint_dir, now.strftime('checkpoint_%Y-%m-%d-%H-%M') + f'_lr_{optimizer.param_groups[0]["lr"]}_batch_{train_loader.batch_size}')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    start_epoch = 0
    best_accuracy = 0.0
    best_loss = float('inf')
    early_stopping_counter = 0

    # Resume from a checkpoint if provided
    if resume_checkpoint_path and os.path.isfile(resume_checkpoint_path):
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint.get('best_accuracy', best_accuracy)
        best_loss = checkpoint.get('best_loss', best_loss)
        losses = np.append(checkpoint['train_loss'], np.empty(num_epochs - len(checkpoint['train_loss'])))
        test_losses = np.append(checkpoint['val_loss'], np.empty(num_epochs - len(checkpoint['val_loss'])))
        train_accuracies = np.append(checkpoint['train_acc'], np.empty(num_epochs - len(checkpoint['train_acc'])))
        test_accuracies = np.append(checkpoint['val_acc'], np.empty(num_epochs - len(checkpoint['val_acc'])))
    else:
        # Initialize arrays to track the losses and accuracies
        losses = np.empty(num_epochs)
        test_losses = np.empty(num_epochs)
        train_accuracies = np.empty(num_epochs)
        test_accuracies = np.empty(num_epochs)
    
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(start_epoch, num_epochs):
        temp = time.time()
        train_loss, train_accuracy = train_epoch(model, optimizer, criterion, train_loader, device)
        test_loss, test_accuracy = test(model, criterion, valid_loader, device)
        
        if scheduler:
                scheduler.step()
                print(f"Epoch {epoch+1}: lr={optimizer.param_groups[0]['lr']}")
                
        losses[epoch] = train_loss
        train_accuracies[epoch] = train_accuracy
        test_accuracies[epoch] = test_accuracy
        test_losses[epoch] = test_loss
        
        # Save checkpoint after every epoch
        if (epoch + 1) % chkpt_interval == 0:
            checkpoint = {
               'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': losses[:epoch+1],
                'train_acc': train_accuracies[:epoch+1],
                'val_loss': test_losses[:epoch+1],
                'val_acc': test_accuracies[:epoch+1]
            }
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
        
        # Save the best model if test accuracy has improved
        if test_accuracies[epoch] > best_accuracy:
            best_accuracy = test_accuracies[epoch]
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
        
        # Print statistics
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"Ep[{epoch+1}/{num_epochs}]  [train loss]: {train_loss:.4f}  "
                  f"[Val Loss]: {test_loss:.4f}  "
                  f"[Best Loss]: {best_loss:.4f}  [TestAcc]: {test_accuracy:.4f}  "
                  f"[Time]: {(time.time() - temp):.2f}")
    
        history = {
            'train_loss': losses[:epoch+1].tolist(),
            'train_acc': train_accuracies[:epoch+1].tolist(),
            'val_loss': test_losses[:epoch+1].tolist(),
            'val_acc': test_accuracies[:epoch+1].tolist()
        }
        # save history to checkpoint_dir as json
        with open(os.path.join(checkpoint_dir, 'history.json'), 'w') as f:
            json.dump(history, f)
            
        # Early stopping check
        if test_losses[epoch] < best_loss:
            best_loss = test_losses[epoch]
            early_stopping_counter = 0  # reset the early stopping counter if the validation loss improves
        else:
            early_stopping_counter += 1  # increment the counter if the validation loss does not improve
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Validation loss did not improve for {early_stopping_patience} consecutive epochs.")
            break
    
    return model, history, test_loader

def plot_training_history(history, history_json_path = None):
    if history_json_path:
        with open(history_json_path) as f:
            history = json.load(f)
            
    plt.figure(figsize=(12, 4))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
        
    
def plot_PR_curve(model, test_loader, device = "cuda:0"):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Range of thresholds to evaluate
    thresholds = np.linspace(0.05, 0.95, 1000)

    # Metrics
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for threshold in thresholds:
        # Make predictions based on the threshold
        predicted = (np.array(all_probs) >= threshold)

        # Calculate metrics
        accuracies.append(accuracy_score(all_labels, predicted))
        precisions.append(precision_score(all_labels, predicted, zero_division=0))
        recalls.append(recall_score(all_labels, predicted, zero_division=0))
        f1_scores.append(f1_score(all_labels, predicted, zero_division=0))
    

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, label='Accuracy')
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.title('Metrics vs Decision Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    return all_probs, all_labels

def plot_confusion_matrix(all_probs, all_labels, threshold=0.5):

    predicted = (np.array(all_probs) >= threshold)
    confusion_matrix_df = confusion_matrix(all_labels, predicted)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_df, annot=True, fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
def test_model(model, test_loader, device = "cuda:0"):
    all_probs, all_labels = plot_PR_curve(model, test_loader, device)
    plot_confusion_matrix(all_probs, all_labels)