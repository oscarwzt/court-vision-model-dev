import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import torchvision
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import datetime
import numpy as np
import json
import cv2
from IPython.display import HTML
from base64 import b64encode
import os
from IPython.display import Video
import subprocess
from pytube import YouTube
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import shutil
import moviepy.editor as mp
import subprocess
import datetime
from yt_dlp import YoutubeDL
import re
    
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



def get_video_info(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        raise ValueError("Error: Could not open the video file.")
    
    # Get the frame width and frame height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fps = cap.get(cv2.CAP_PROP_FPS)

    return cap, fps, frame_width, frame_height

def display_video(input_path, width=640, ffmpeg_path='ffmpeg-git-20231128-amd64-static/ffmpeg'):
    temp_output_path = "temp_" + os.path.basename(input_path)

    try:
        # Use subprocess to safely call FFmpeg
        subprocess.run([ffmpeg_path, '-y', '-i', input_path, '-vcodec', 'libx264', temp_output_path],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,
                       check=True)

        # Overwrite the original file with the compressed one
        shutil.move(temp_output_path, input_path)

        with open(input_path, 'rb') as file:
            mp4 = file.read()
        data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

        # Display video in HTML
        display_html = f"""
        <video width={width} controls>
            <source src="{data_url}" type="video/mp4">
        </video>
        """
        return HTML(display_html)



    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        # Clean up the temporary file in case of an error
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)

def generateHighlight(video_path,
                      score_timestamps, 
                      clip_start_offset = 6, # number of seconds before scoring
                      clip_end_offset = 3,   # number of seconds after scoring
                      output_path = "videos_clipped/scored"):
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Calculate clip lengths in frames
    start_frame_offset = clip_start_offset * fps
    end_frame_offset = clip_end_offset * fps

    # For each score event
    for i, timestamp in enumerate(score_timestamps):
        print(f'Processing clip {i}')
        # Calculate start and end frames for this clip
        score_frame = np.floor(timestamp * fps)
        start_frame = int(max(0, score_frame - start_frame_offset))
        end_frame = min(total_frames - 1, np.ceil(score_frame + end_frame_offset))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        video_path = f'{output_path}/clip_{i}.mp4'
        out = cv2.VideoWriter(video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        # Copy frames from the input video to the output clip
        for _ in tqdm(range(start_frame, end_frame + 1)):

            ret, frame = cap.read()

            if ret:
                out.write(frame)
            else:
                break

        # Close the output clip
        out.release()

    # Close the input video
    cap.release()
    
    
def download_video(url, save_path, resolution=None):
    yt = YouTube(url)
    if resolution:
        video = yt.streams.filter(mime_type="video/mp4", res = resolution).first()
    else:
        video = yt.streams.filter(mime_type="video/mp4").order_by("resolution").desc().first()

    # Reformat the video name
    video_name = video.default_filename.replace(" ", "").replace("/", "_").replace("-", "_")
    # add current date to the video_name in the format of "YYYY_MM_DD_video_name"
    video_name = datetime.datetime.now().strftime("%Y_%m_%d_") + '_' + video_name 
    
    # Split the name and the extension
    name_part, ext_part = os.path.splitext(video_name)

    # Remove non-alphanumeric and non-underscore characters from the name part
    name_part = re.sub(r'\W+', '', name_part)

    # Join the name part and the extension part
    video_name = name_part + ext_part
    video_file_path = os.path.join(save_path, video_name)
    
    # if video does not exist, download it
    if not os.path.isfile(os.path.join(save_path, video_name)):
        print(f'Downloading video {video_name}...')
        video.download(output_path=save_path, filename=video_name)
    else:
        print(f'Video {video_name} already exists.')

    # # If the downloaded video is in WebM format, convert it to MP4 using FFmpeg
    # if ext_part.lower() == '.webm' and not os.path.isfile(os.path.splitext(video_file_path)[0] + '.mp4'):
    #     mp4_output_path = os.path.splitext(video_file_path)[0] + '.mp4'
    #     print("converting")
    #     subprocess.run(['ffmpeg', '-i', video_file_path, '-c:v', 'libx264', '-c:a', 'aac', mp4_output_path], check=True)
    #     os.remove(video_file_path)  # Remove the original WebM file

    #     return mp4_output_path

    return video_file_path

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
