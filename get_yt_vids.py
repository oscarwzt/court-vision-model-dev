import cv2
import os
from os import listdir
import random
import re
from pytube import YouTube
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import shutil
import moviepy.editor as mp
import subprocess

def download_video(url, save_path, resolution=None):
    yt = YouTube(url)
    if resolution:
        video = yt.streams.filter(res = resolution).first()
    else:
        video = yt.streams.filter(mime_type="video/mp4").order_by("resolution").desc().first()

    # Reformat the video name
    video_name = video.default_filename.replace(" ", "").replace("/", "_").replace("-", "_").replace(".", "_")
    video_name = "id_" + url.split("v=")[1] + ".mp4" if len(video_name) > 20 else video_name
    
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

    # If the downloaded video is in WebM format, convert it to MP4 using FFmpeg
    if ext_part.lower() == '.webm' and not os.path.isfile(os.path.splitext(video_file_path)[0] + '.mp4'):
        mp4_output_path = os.path.splitext(video_file_path)[0] + '.mp4'
        print("converting")
        subprocess.run(['ffmpeg', '-i', video_file_path, '-c:v', 'libx264', '-c:a', 'aac', mp4_output_path], check=True)
        os.remove(video_file_path)  # Remove the original WebM file

        return mp4_output_path

    return video_file_path

#### cut downloaded video to specified interval to test model performance ####
def cut_video(video_path, output_path,start_time, end_time, fps):
    # Get the file extension of the input video
    _, file_extension = os.path.splitext(video_path)

    # Load the video clip
    clip = mp.VideoFileClip(video_path).subclip(start_time, end_time)

    # Choose the appropriate codec based on the file extension
    codec = 'libx264' if file_extension == '.mp4' else 'libvpx-vp9'  # For .mp4 use H.264, for others use VP9

    # Write the trimmed video to the output file with the selected codec
    clip.write_videofile(output_path, codec=codec, fps = fps)
    
def extract_frames(video_path, frames_dir, num_frames, total_seconds, start_time, end_time):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the frame indices for the specified start and end times
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Generate a list of all frame indices
    all_frames = list(range(int(total_seconds * fps)))

    # Remove the frame indices that fall into the specified interval
    available_frames = [f for f in all_frames if f < start_frame or f >= end_frame]

    # Randomly select frame indices from the available frames
    frame_indices = random.sample(available_frames, num_frames)

    # Create a separate folder for each video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_frames_dir = os.path.join(frames_dir, video_name)
    os.makedirs(video_frames_dir, exist_ok=True)

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"Frame at position {frame_idx} could not be read.")
            continue

        # Save the frame within the video-specific folder
        frame_name = f"frame_{video_name}_{i}.jpg"
        cv2.imwrite(os.path.join(video_frames_dir, frame_name), frame)

    
def process_video(video_url = None, 
                  video_local_path = None, # input a local video
                  clip_video=False,     # whether to save a clipped version (for model testing, et.)
                  save_full_video=False,# whether to save the full video
                  resolution = None,    # format: '1440p', '1080p', '720p' ...
                  video_save_path='./videos_full',
                  clip_save_path='./videos_clipped', 
                  frames_dir='./yt_frames',  # directory to store video frames (for model training)
                  num_frames=20,        # number of frames to extract
                  interval_length=8):

    # Download video
    print("downloading video")
    video_path = video_local_path if video_local_path else download_video(video_url, video_save_path, resolution) 
    

    # Get video properties
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_seconds = int(total_frames / cap.get(cv2.CAP_PROP_FPS))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    
    # Calculate the available duration for the interval
    available_duration = total_seconds - interval_length

    if available_duration <= 0:
        print("Video duration is shorter than the specified interval.")
        return

    # Generate the random start time for the interval
    start_time = random.randint(0, available_duration)

    # Calculate the end time based on the start time and interval length
    end_time = start_time + interval_length
    
    # Extract frames before trimming the video
    print("extracting frames from {} to {}".format(start_time, end_time))
    
    # skip if frames already exist
    if os.path.isdir(os.path.join(frames_dir, os.path.splitext(os.path.basename(video_path))[0])):
        print("frames already exist")
    else:
        extract_frames(video_path, frames_dir, num_frames, total_seconds, start_time, end_time)

    # Clip the video if requested
    if clip_video:
        print("clipping video")
        trimmed_video_path = os.path.join(clip_save_path, f"{os.path.splitext(os.path.basename(video_path))[0]}_trimmed.mp4")
        cut_video(video_path, trimmed_video_path, start_time, end_time, fps)

    # Save the whole video if requested
    if not save_full_video and not video_local_path:
        os.remove(video_path)
