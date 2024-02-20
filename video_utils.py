import os
import datetime
import cv2
from IPython.display import HTML
from base64 import b64encode
import subprocess
from pytube import YouTube
import shutil
import re
from tqdm import tqdm
import numpy as np

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

def display_video(input_path, width=640, ffmpeg_path='ffmpeg/ffmpeg'):
    # get input_path's directory
    input_dir = os.path.dirname(input_path)
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
            

def initialize_video_writer(fps, video_dimension, video_path, output_dir = None, saved_video_name = None, codec="mp4v"):
    video_name = video_path.split("/")[-1]
    video_name = video_name.split(".")[0] + ".mp4"

    if saved_video_name is not None:
        output_path = saved_video_name if output_dir is None else os.path.join(output_dir, saved_video_name)
    else:
        output_path = "inferenced_" + video_name if output_dir is None else os.path.join(output_dir, "inferenced_" + video_name)
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, codec, fps, video_dimension)
    return out, output_path

def initialize_video_capture(video_path, skip_to_sec = 0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if skip_to_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, skip_to_sec * 1000)
        
    num_skiped_frames = int(skip_to_sec * fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - num_skiped_frames
    return cap, fps, frame_width, frame_height, total_frames

def trim_highlights_from_timestamps(video_path,
                      score_timestamps, 
                      clip_start_offset = 6, # number of seconds before scoring
                      clip_end_offset = 2,   # number of seconds after scoring
                      output_path = ".",
                      ffmpeg_path = "ffmpeg-git-20240203-amd64-static/ffmpeg"):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    video_name = os.path.basename(video_path)
    
    for i, timestamp in enumerate(score_timestamps):
        start_time = max(0, timestamp - clip_start_offset)
        end_time = timestamp + clip_end_offset
        clip_output_path = os.path.join(output_path, f"{video_name}_highlight_{i}.mp4")

        # Construct FFmpeg command for trimming
        ffmpeg_command = [ffmpeg_path, '-i', video_path, '-ss', str(start_time), '-to', str(end_time), '-c', 'copy', clip_output_path]
        subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    



