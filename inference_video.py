from model import MODEL
import cv2
from video_utils import *
import torch
import math
import numpy as np
import time
from obj_detection_utils import *
from get_yt_vids import *

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = MODEL("weights/best.pt", device)

def predict_video(video_path, 
                  model = model,
                  batch_size = 32,
                  skip_to_sec = 0,
                  end_at_sec = -1,
                  show_progress = False,
                  write_video = False,
                  save_output_video_name = "output.mp4",
                  ):
    """
    Predicts the score of a basketball video based on ball detection and tracking.

    Parameters:
    - video_path (str): The path to the input video file.
    - model (object): The ball detection model to use for prediction.
    - batch_size (int): The number of frames to process in each batch.
    - skip_to_sec (float): The number of seconds to skip at the beginning of the video.
    - end_at_sec (float): The number of seconds to stop processing the video. (default: -1, process the entire video)
    - show_progress (bool): Whether to display a progress bar during processing.
    - write_video (bool): Whether to write the output video with annotations.
    - save_output_video_name (str): The name of the output video file.

    Returns:
    - timestamps (list): A list of timestamps where a score was detected.
    """
    
    cap, fps, frame_width, frame_height, total_frames = initialize_video_capture(video_path=video_path, skip_to_sec=skip_to_sec)
    if write_video:
        out, output_path = initialize_video_writer(fps = fps,
                                                   video_dimension= (frame_width, frame_height),
                                                   video_path=video_path,
                                                   saved_video_name=save_output_video_name
                                                   )
    num_batches = math.ceil(total_frames / batch_size)
    reached_stopping_time = False
    box_containing_ball_prev = None
    box_containing_ball_cur = None
    score = 0
    last_scored_time = -1
    
    timestamps = []
    
    if show_progress:
        batch_range = tqdm(range(num_batches), desc='Processing Batches')
    else:
        batch_range = range(num_batches)

    for i in batch_range:
        start_time = time.time()  # Start time for fps calculation
        if reached_stopping_time:
            break
        frames = []
        for _ in range(batch_size):
            ret, img = cap.read()
            if ret:
                frames.append(img)
            else:
                break

        if frames:
            results = model.predict(frames)
        else:
            continue

        for idx, (frame, r) in enumerate(zip(frames, results)):
            current_frame_num = idx + i * batch_size
            current_time = skip_to_sec + current_frame_num / fps
            if current_time >= end_at_sec and end_at_sec > 0:
                reached_stopping_time = True
                break
            
            if write_video:
                cv2.putText(frame, f"Score: {score}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)

            bounding_boxes = r['boxes']
            labels = r['labels']

            labels = [model.model.names[i] for i in labels]
            objects = {label: [] for label in labels}
            

            for box, label in zip(bounding_boxes, labels):
                objects[label].append(box)
            
            if "basketball" not in objects or "hoop" not in objects:
                if write_video:
                    out.write(frame)       
                continue
            hoop_boxes = objects["hoop"]
            detection_areas = [get_detection_box(*box) for box in hoop_boxes]
            entry_boxes = [get_entry_box(*box) for box in hoop_boxes]
            exit_boxes = [get_exit_box(*box) for box in hoop_boxes]
            relevant_ball_boxes = [box for box in objects["basketball"] 
                                            for det_area in detection_areas
                                            if is_in_box(*box, *det_area)]
            if not relevant_ball_boxes:
                if write_video:
                    out.write(frame)
                continue

            if write_video:
                for ball_boxes in objects["basketball"]:
                    cv2.circle(frame, get_center(*ball_boxes), 5, COLORS["basketball"], -1)
            focus_areas = {
                #"detection_area": detection_areas,
                "hoop_box": hoop_boxes,
                "entry_box": entry_boxes,
                "exit_box": exit_boxes
            }
            
            # determine which box the ball is in
            for box_name, all_boxes in focus_areas.items():
                for box in all_boxes:
                    if any([is_in_box(*relevant_ball_boxes, *box, threshold=0.55) for relevant_ball_boxes in relevant_ball_boxes]):
                        box_containing_ball_cur = box_name #if not no_relevant_ball else None
                        if write_video:
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), COLORS[box_name], 2)
                            cv2.putText(frame, box_name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[box_name], 2)
                    else:
                        if write_video:
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 0), 2)
                            cv2.putText(frame, box_name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            ball_in_interested_area = (box_containing_ball_cur == "hoop_box" or box_containing_ball_cur == "exit_box")
            
            time_since_last_scored = current_time - last_scored_time
            if box_containing_ball_prev == "entry_box" and ball_in_interested_area and time_since_last_scored > 0.5:
                score += 1
                last_scored_time = current_time
                timestamps.append(current_time)
                
            box_containing_ball_prev = box_containing_ball_cur
            if write_video:
                cv2.putText(frame, f"Score: {score}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
                cv2.putText(frame, f"ball in: {box_containing_ball_cur}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
                out.write(frame)  
        if show_progress:
            elapsed_time = time.time() - start_time  # Elapsed time for batch
            fps = batch_size / elapsed_time  # Calculate fps based on batches processed
            batch_range.set_postfix(fps=f"{fps:.2f} fps", refresh=True)        
         
    cap.release()
    if write_video:
        out.release()
        print(f"Output video saved at {output_path}")
    
    return timestamps