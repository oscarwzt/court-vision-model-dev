import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt


def load_checkpoint(model, checkpoint_path, device = "cuda", num_classes=1 ):
    """
    Load a model checkpoint and restore the model and optimizer states.

    Parameters:
    - model: The model instance on which to load the parameters.
    - optimizer: The optimizer instance for which to restore the state.
    - checkpoint_path: The path to the checkpoint file.

    Returns:
    - model: The model with restored parameters.
    - optimizer: The optimizer with restored state.
    - epoch: The epoch at which the checkpoint was saved.
    - loss: The loss value at the checkpoint.
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location = device)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
      
def load_resnet18(cls_model_chkpoint_path, num_classes = 1, device = "cuda"):
    cls_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)    
    load_checkpoint(cls_model, checkpoint_path = cls_model_chkpoint_path, device = device)
    cls_model.to(device)
    cls_model.eval()  
    return cls_model

def load_resnet50(cls_model_chkpoint_path, num_classes = 1, device = "cuda"):
    cls_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)    
    load_checkpoint(cls_model, checkpoint_path = cls_model_chkpoint_path, device = device)
    cls_model.to(device)
    cls_model.eval()  
    return cls_model
    
def cls_predict_image(cls_model, img, preprocess, device, threshold = 0.5):
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        cls_output = cls_model(input_tensor)
    probability = torch.sigmoid(cls_output.squeeze())

    # prob, predicted_class = torch.max(probability, dim=0)
    # return predicted_class.item(), prob.item()
    return probability[1] > threshold, probability

def predict_hoop_box(img, cls_model, x1, y1, x2, y2, preprocess, device, threshold = 0.5):
    cropped_img = img[y1:y2, x1:x2]
    cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
    # Preprocess the cropped image
    predicted_class, prob = cls_predict_batch(cls_model, cropped_img_pil, preprocess, device, threshold)
    return cropped_img_pil, predicted_class, prob

def cls_predict_batch(cls_model, batch_imgs, preprocess, device, threshold = 0.5):
    # Process and batch images
    # check if batch_imgs is a list.
        
    if isinstance(batch_imgs, list):
        batch_tensor = torch.stack([preprocess(img) for img in batch_imgs])
        batch_tensor = batch_tensor.to(device)
    else:
        batch_tensor = preprocess(batch_imgs).unsqueeze(0).to(device)

    # Forward pass for the whole batch
    with torch.no_grad():
        cls_output = cls_model(batch_tensor)

    # Calculate probabilities and predicted classes
    probabilities = torch.sigmoid(cls_output)
    return probabilities > threshold, probabilities.tolist()

def predict_hoop_box_batch(img, cls_model,  preprocess, device, threshold = 0.5):
    cropped_imgs_pil = []

    for cropped_img in img:
        cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
        cropped_imgs_pil.append(cropped_img_pil)

    return cls_predict_batch(cls_model, cropped_imgs_pil, preprocess, device, threshold)

def show_feature_map(model, image_path, device = "cuda"):
    img = Image.open(image_path)

    feature_maps = []

    def hook_function(module, input, output):
        feature_maps.append(output)

    # Register the hook to the first convolutional layer of the first block of layer1
    hook = model.layer1[0].conv1.register_forward_hook(hook_function)

    # Load an image


    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor(),
    ])
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model(img_tensor)

    # Unregister the hook
    hook.remove()

    # Visualize the feature maps
    fm = feature_maps[0].squeeze(0)  # Get the feature maps

    # Plotting
    fig, axes = plt.subplots(8, 8, figsize=(20, 20))  # Adjust subplot dimensions as needed
    for i, ax in enumerate(axes.flat):
        if i < 64:  # Adjust this based on how many feature maps you want to visualize
            ax.imshow(fm[i].detach().cpu().numpy(), cmap='viridis')
            ax.axis('off')
    plt.tight_layout()
    plt.show()