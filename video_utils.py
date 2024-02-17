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

def display_video(input_path, width=640, ffmpeg_path='ffmpeg-git-20240203-amd64-static/ffmpeg'):
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

def generateHighlight(video_path,
                      score_timestamps, 
                      clip_start_offset = 6, # number of seconds before scoring
                      clip_end_offset = 2,   # number of seconds after scoring
                      output_path = ".",
                      ffmpeg_path = "ffmpeg-git-20240203-amd64-static/ffmpeg"):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for i, timestamp in enumerate(score_timestamps):
        start_time = max(0, timestamp - clip_start_offset)
        end_time = timestamp + clip_end_offset
        clip_output_path = os.path.join(output_path, f"highlight_{i}.mp4")

        # Construct FFmpeg command for trimming
        ffmpeg_command = [ffmpeg_path, '-i', video_path, '-ss', str(start_time), '-to', str(end_time), '-c', 'copy', clip_output_path]
        subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    
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


