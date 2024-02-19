import cv2
import os
from os import listdir
import random
from numpy import save
from pytube import YouTube


import subprocess
import argparse
from urllib.parse import urlparse, parse_qs
from yt_dlp import YoutubeDL


def get_yt_video_id(url):
    parsed_url = urlparse(url)
    if 'youtu.be' in parsed_url.netloc:
        # Extract the video ID from the path for shortened URLs
        video_id = parsed_url.path.lstrip('/')
    elif 'youtube.com' in parsed_url.netloc:
        # Extract the video ID from the query for standard URLs
        query_string = parse_qs(parsed_url.query)
        video_id = query_string.get('v', [None])[0]
    else:
        video_id = None
        print("Invalid YouTube URL")
    return video_id

def download_video(url, save_path, resolution=None):
    yt = YouTube(url)
    if resolution:
        video = yt.streams.filter(mime_type="video/mp4", res = resolution).first()
    else:
        video = yt.streams.filter(mime_type="video/mp4").order_by("resolution").desc().first()

    video_id = get_yt_video_id(url)
    video_path = os.path.join(save_path, f"{video_id}.mp4")
    
    # if video does not exist, download it
    if not os.path.isfile(video_path):
        print(f'Downloading video {video_id}...')
        video.download(output_path=save_path, filename=video_id)
    else:
        print(f'Video {video_id} already exists.')
    return video_path


def download_video(url, save_path, resolution=None, ffmpeg_path = "ffmpeg-git-20240203-amd64-static/ffmpeg"):
    # Define download options for yt-dlp
    ydl_opts = {
        'outtmpl': os.path.join(save_path, '%(id)s.%(ext)s'),
        'format': 'bestvideo',
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',  # Convert to mp4 if necessary
            "ffmpeg_location": f"{ffmpeg_path}"
        }],  
    }
    
    # If a specific resolution is requested, adjust the format selection
    if resolution:
        ydl_opts['format'] = f'bestvideo[height<={resolution}]'
    else:
        # Ensure the format is set to mp4 for consistency and compatibility
        ydl_opts['format'] += '[ext=mp4]'

    # Ensure the save directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    video_id = get_yt_video_id(url)
    video_path = os.path.join(save_path, f"{video_id}.mp4")
    if os.path.isfile(video_path):
        print(f'Video {video_id} already exists.')
        return video_path
    # Download the video
    with YoutubeDL(ydl_opts) as ydl:
        video_info = ydl.extract_info(url, download=True)
        video_id = video_info.get('id')
        
    return video_path

    
def extract_frames(video_path, frames_dir, num_frames):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_frame_indicies = list(range(total_frames))
    frame_indices = random.sample(all_frame_indicies, num_frames)

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
        frame_name = f"{video_name}_{i:02d}.jpg"
        cv2.imwrite(os.path.join(video_frames_dir, frame_name), frame)

    
def process_video(video_url = None, 
                  video_local_path = None, # input a local video
                  save_full_video=False,# whether to save the full video
                  resolution = None,    # format: '1440', '1080', '720' ...
                  video_save_path='./videos_full',
                  frames_dir='./yt_frames',  # directory to store video frames (for model training)
                  num_frames=20,        # number of frames to extract
                  ):

    # Download video
    print("downloading video")
    video_path = video_local_path if video_local_path else download_video(video_url, video_save_path, resolution) 
    
    # skip if frames already exist
    if os.path.isdir(os.path.join(frames_dir, os.path.splitext(os.path.basename(video_path))[0])):
        print("frames already exist")
    else:
        extract_frames(video_path, frames_dir, num_frames)

    # Save the whole video if requested
    if not save_full_video and not video_local_path:
        os.remove(video_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a video.')
    
    parser.add_argument('--video_url', type=str, help='URL of the video to process')
    parser.add_argument('--video_local_path', type=str, default=None, help='Path to a local video file to process')
    parser.add_argument('--save_full_video', action='store_true', help='Whether to save the full video')
    parser.add_argument('--resolution', type=str, default=None, help="Video resolution (e.g., '1440p', '1080p') default to highest quality")
    parser.add_argument('--video_save_path', type=str, default='./videos_full', help='Path where the full video will be saved')
    parser.add_argument('--frames_dir', type=str, default='./yt_frames', help='Directory to store video frames')
    parser.add_argument('--num_frames', type=int, default=20, help='Number of frames to extract')

    args = parser.parse_args()
    
    # Call your function with the parsed arguments
    process_video(
        video_url=args.video_url,
        video_local_path=args.video_local_path,
        save_full_video=args.save_full_video,
        resolution=args.resolution,
        video_save_path=args.video_save_path,
        frames_dir=args.frames_dir,
        num_frames=args.num_frames
    )