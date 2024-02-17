import cv2
import torch
import numpy as np
import math
from numpy import random
import os
from video_utils import *
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from IPython.display import display
from IPython.display import Video
from video_utils import display_video

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
])

def convert_images_to_tensor(img_list, preprocess):
    if isinstance(img_list, np.ndarray):
        img_list = [img_list]
    tensor_list = [preprocess(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in img_list]
    stacked_tensor = torch.stack(tensor_list, dim=0)
    return stacked_tensor

def model_predict(cls_model, batch_imgs, device, threshold=0.5):
    # Move the batch to the specified device
    batch_imgs = batch_imgs.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = cls_model(batch_imgs)

    # Apply sigmoid to output probabilities
    probabilities = torch.sigmoid(outputs)

    # Convert probabilities to binary predictions based on the threshold
    predictions = probabilities >= threshold

    return predictions.flatten(), probabilities.flatten()

def predict_hoop_box(img_list, cls_model, preprocess, device, threshold=0.5):
    batch_imgs = convert_images_to_tensor(img_list, preprocess)
    predictions, probabilities = model_predict(cls_model, batch_imgs, device, threshold)
    return predictions.cpu().numpy(), probabilities.cpu().numpy()

    

def inference_by_batch(model,
                       cls_model,
                       video_path, 
                       cls_conf_threshold = 0.6,
                       detect_conf_threshold = 0.4,
                       save_result_vid = False, 
                       output_dir = None, 
                       saved_video_name = None,
                       batch_size=128,
                       display_result = False,
                       show_progress = True,
                       skip_to_sec = 0,
                       show_score_prob = False,
                       cls_img_size = 112,
                       device = device,
                       ):
    cap, fps, frame_width, frame_height, total_frames = initialize_video_capture(video_path, skip_to_sec)
    if save_result_vid:
        out, output_path = initialize_video_writer(fps, (frame_width,frame_height), video_path, output_dir, saved_video_name)
        
    num_batches = math.ceil(total_frames / batch_size)

    results = []
    score_timestamps = []
    
    count = 61
    score = 0
    display_prob = [0.0]
    
    if show_progress:
        batch_range = tqdm(range(num_batches))
    else:
        batch_range = range(num_batches)

    for i in batch_range:
        frames = []
        for i in range(batch_size):
            ret, img = cap.read()
            if ret:
                frames.append(img)
            else:
                break

        if frames:
            results = model(frames, 
                            stream=False, 
                            verbose = False, 
                            conf=detect_conf_threshold,
                            device=device)
        else:
            continue
        

        for c, r in tqdm(enumerate(results)):
            #print(c)
            img = r.orig_img
            boxes = r.boxes
            cropped_images = []
            count += 1
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # convert to int values
                confidence = box.conf.item()
                predicted_class = model.names[int(box.cls)] 
                if predicted_class == "hoop":
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(img, f'{predicted_class}: {confidence:.3f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    if x1 > x2 or y1 > y2:
                        continue
                    else:
                        cropped_img = img[y1:y2, x1:x2]
                        cropped_images.append(cropped_img)
                        
                # if predicted_class == "basketball":
                #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #     cv2.putText(img, f'{predicted_class}: {confidence:.3f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
            
            if len(cropped_images) == 0:
                continue
            pred, prob = predict_hoop_box(cropped_images, cls_model,  preprocess, device, threshold=cls_conf_threshold)
            if pred.sum() > 0 and count > 60:
                score += 1
                count = 0
                current_frame = i * batch_size + c
                time_stamp = current_frame / fps
                score_timestamps.append((time_stamp, prob))
                display_prob = prob
        
            cv2.putText(img, f'Score: {score}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            # if show_score_prob:
            #     cv2.putText(img, f'Prob: {max(display_prob):.3f}', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            if save_result_vid:
                out.write(img)
        print("finished inferencing with cls")
        if not ret:
            break
        
    if save_result_vid:
        out.release()
    cap.release()
    if display_result:
        display_video(output_path, ffmpeg_path="ffmpeg")
        return score_timestamps, output_path
    else:
        return score_timestamps
    
    
def inference_by_frame(model, 
                       cls_model,
                       video_path, 
                       cls_conf_threshold = 0.6,
                       detect_conf_threshold = 0.4,
                       save_result_vid = False, 
                       output_dir = None, 
                       saved_video_name = None,
                       display_result = False,
                       show_progress = True,
                       skip_to_sec = 0,
                       show_score_prob = False,
                       device = device,
                       preprocess = preprocess
                       ):

    
    cap, fps, frame_width, frame_height = get_video_info(video_path)
    if skip_to_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, skip_to_sec * 1000)
        
    num_skiped_frames = int(skip_to_sec * fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - num_skiped_frames
    
    
    if save_result_vid:
        video_name = video_path.split("/")[-1]
        video_name = video_name.split(".")[0] + ".mp4"

        if saved_video_name is not None:
            output_path = saved_video_name if output_dir is None else os.path.join(output_dir, saved_video_name)
        else:
            output_path = "inferenced_" + video_name if output_dir is None else os.path.join(output_dir, "inferenced_" + video_name)
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, codec, fps, (frame_width,frame_height))
    pbar = tqdm(total=total_frames, desc="Processing Frames", unit="frame") if show_progress else None

    score_timestamps = []
    count=61
    score = 0
    display_prob = 0.0
    while True:
        ret, img = cap.read()
        frame_start_time = time.time()
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        count += 1
        if ret:
            results = model(img, stream = False, device = device, conf = detect_conf_threshold, verbose = False)
            
            for r in results:
                boxes = r.boxes

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                    confidence = box.conf[0]
                    predicted_class = model.names[int(box.cls)]
                    
                    # If "basketball-hoops" is detected, make a prediction with cls_model
                    if predicted_class == "hoop":
                        # Crop the image and convert to PIL Image
                        # try:
                        if x1 > x2 or y1 > y2:
                            continue
                        else:
                            cropped_img = img[y1:y2, x1:x2]
                            prediction, prob = predict_hoop_box([cropped_img], cls_model, preprocess, device, cls_conf_threshold)
                            #print(prediction)
                            if any(prediction == 1) and count > 60:
                                score += 1
                                count = 0
                                display_prob = prob[0]
                                score_timestamps.append((current_time, prob))

                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(img, f'{predicted_class}: {confidence:.3f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    # if predicted_class == "basketball":
                    #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    #     cv2.putText(img, f'{predicted_class}: {confidence:.3f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(img, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            if show_score_prob:
                # print(display_prob)
                cv2.putText(img, f'Prob: {display_prob:.3f}', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

            if show_progress:
                frame_end_time = time.time()  # End time for frame processing
                time_per_frame = frame_end_time - frame_start_time
                pbar.set_postfix(time_per_frame=f"{time_per_frame:.3f} sec")
                pbar.update(1)
            
        else:
            break
        
        if save_result_vid:
            out.write(img)
            
    if save_result_vid:
        out.release()
    cap.release()
    if display_result:
        display_video(output_path, ffmpeg_path="ffmpeg")
        return score_timestamps, output_path
    else:
        return score_timestamps

def inference_by_batch_(frames,
                       model,
                       cls_model,
                       preprocess,
                       cls_conf_threshold = 0.6,
                       detect_conf_threshold = 0.4,
                       ):
    if len(frames) == 0:
        return 
    results = model(frames,
                    stream=False,
                    verbose=False,
                    conf=detect_conf_threshold,
                    device=device)
    predictions = []
    for r in results:
        img = r.orig_img
        boxes = r.boxes
        cropped_images = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf.item()
            predicted_class = model.names[int(box.cls)]
            if predicted_class == "hoop":
                if x1 > x2 or y1 > y2:
                    continue
                else:
                    cropped_img = img[y1:y2, x1:x2]
                    cropped_images.append(cropped_img)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, f'{predicted_class}: {confidence:.3f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    
        if len(cropped_images) == 0:
            continue
        pred = predict_hoop_box(cropped_images, cls_model, preprocess, device, threshold=cls_conf_threshold)
        predictions.append(pred)
        
    return predictions

def inference_by_frame_(model,
                       cls_model,
                       cap, 
                       score_timestamps,
                       show_progress = True,
                       detect_conf_threshold = 0.4,
                       cls_conf_threshold = 0.5,
                       show_score_prob = False,
                       video_writer = None,
                       device = "cuda"
                       ):
    count = 61
    score = 0
    display_prob = 0.00
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_range = tqdm(range(total_frames)) if show_progress else range(total_frames)
    for i in frame_range:
        ret, img = cap.read()
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        count += 1
        if ret:
            results = model(img, stream = False, device = device, conf = detect_conf_threshold, verbose = False)
            
            for r in results:
                boxes = r.boxes

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                    confidence = box.conf[0]
                    predicted_class = model.names[int(box.cls)]
                    
                    # If "basketball-hoops" is detected, make a prediction with cls_model
                    if predicted_class == "hoop":
                        # Crop the image and convert to PIL Image
                        # try:
                        if x1 > x2 or y1 > y2:
                            continue
                        else:
                            cropped_img = img[y1:y2, x1:x2]
                            prediction, prob = predict_hoop_box([cropped_img], cls_model, preprocess, device, cls_conf_threshold)
                            #print(prediction)
                            if any(prediction == 1) and count > 60:
                                score += 1
                                count = 0
                                display_prob = prob[0]
                                current_time = round(current_time / 1000, 2)
                                score_timestamps.append((current_time, prob))

                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(img, f'{predicted_class}: {confidence:.3f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    if predicted_class == "basketball":
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, f'{predicted_class}: {confidence:.3f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                    if predicted_class == "made":
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(img, f'{predicted_class}: {confidence:.3f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                    if predicted_class == "person":
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(img, f'{predicted_class}: {confidence:.3f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.putText(img, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            if show_score_prob:
                # print(display_prob)
                cv2.putText(img, f'Prob: {display_prob:.3f}', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                
            if video_writer is not None:
                video_writer.write(img)
        else:
            break

def inference_video(model, 
                    cls_model,
                    video_path, 
                    cls_conf_threshold=0.6,
                    detect_conf_threshold=0.4,
                    save_result_vid=False, 
                    output_dir=None, 
                    saved_video_name=None,
                    display_result=False,
                    show_progress=True,
                    skip_to_sec=0,
                    show_score_prob=False,
                    device=None,
                    preprocess=preprocess,
                    batch_mode=False,
                    batch_size=128,
                    ffmpeg_path="ffmpeg-git-20231128-amd64-static/ffmpeg",
                    num_buffer_frames = 20
                    ):
    cap, fps, frame_width, frame_height, total_frames = initialize_video_capture(video_path, skip_to_sec)
    if save_result_vid:
        out, output_path = initialize_video_writer(fps, (frame_width,frame_height), video_path, output_dir, saved_video_name)
        
    buffer_between_made = num_buffer_frames
    score = 0
    score_timestamps = []
    display_prob = 0.0
    
    if batch_mode:
        num_batches = math.ceil(total_frames / batch_size)
        batch_range = tqdm(range(num_batches)) if show_progress else range(num_batches)
        for i in batch_range:
            frames = []
            for _ in range(batch_size):
                ret, img = cap.read()
                if ret:
                    frames.append(img)
                else:
                    break
                
            if frames:
                predictions = inference_by_batch_(frames,
                                                model,
                                                cls_model,
                                                preprocess,
                                                cls_conf_threshold,
                                                detect_conf_threshold)
                for frame_count, (pred_, frame) in enumerate(zip(predictions, frames)):
                    buffer_between_made += 1
                    pred, prob = pred_
                    if any(pred == 1) and buffer_between_made > num_buffer_frames:
                        score += 1
                        buffer_between_made = 0
                        scoring_time = i * batch_size + frame_count
                        scoring_time = round(scoring_time / fps, 2)
                        score_timestamps.append((scoring_time, prob))
                        display_prob = max(prob)
                        
                    cv2.putText(frame, f'Score: {score}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                    if show_score_prob:
                        cv2.putText(frame, f'Prob: {display_prob:.3f}', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                    if save_result_vid:
                        out.write(frame)
            else:
                continue
        
    else:
        print("inference by frame")
        inference_by_frame_(model, 
                            cls_model,
                            cap, 
                            score_timestamps,
                            show_progress,
                            detect_conf_threshold,
                            cls_conf_threshold,
                            show_score_prob,
                            out,
                            device)
        
        

    if save_result_vid:
        out.release()
    cap.release()
    if display_result:
        display_video(output_path, ffmpeg_path=ffmpeg_path)
    return score_timestamps
    
