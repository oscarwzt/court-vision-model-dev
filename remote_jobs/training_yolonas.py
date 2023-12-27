import torch
from super_gradients.training import models
import os
import numpy as np
import math
from numpy import random
import cv2
from super_gradients.training import Trainer
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, coco_detection_yolo_format_val)
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050, DetectionMetrics_050_095
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
import logging
import supervision as sv
from onemetric.cv.object_detection import ConfusionMatrix
from super_gradients.training.utils.distributed_training_utils import setup_device
num_gpus = torch.cuda.device_count()
setup_device(num_gpus=num_gpus)
print("Num GPUs Available: ", torch.cuda.device_count())
logging.getLogger("super_gradients").setLevel(logging.WARNING)

LOCATION = "basketballDetection-21"
CLASSES = ['basketball', 'hoop', 'person']
CHECKPOINT_DIR = 'yolonas_checkpoints'
NUM_WORKERS = 12
BATCH_SIZE = 32
MAX_EPOCHS = 300
EXPERIMENT_NAME = "bball_detect_v21_newconfig"
trainer = Trainer(experiment_name=EXPERIMENT_NAME, ckpt_root_dir=CHECKPOINT_DIR)


dataset_params = {
    'data_dir': LOCATION,
    'train_images_dir':'train/images',
    'train_labels_dir':'train/labels',
    'val_images_dir':'val/images',
    'val_labels_dir':'val/labels',
    'test_images_dir':'test/images',
    'test_labels_dir':'test/labels',
    'classes': CLASSES
}

train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['train_images_dir'],
        'labels_dir': dataset_params['train_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size': BATCH_SIZE,
        'num_workers': NUM_WORKERS
    }
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['val_images_dir'],
        'labels_dir': dataset_params['val_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size': BATCH_SIZE,
        'num_workers': NUM_WORKERS
    }
)

test_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['test_images_dir'],
        'labels_dir': dataset_params['test_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size': BATCH_SIZE,
        'num_workers': NUM_WORKERS
    }
)

train_data.dataset.transforms.pop(2)


model = models.get('yolo_nas_l',
                   num_classes=len(dataset_params['classes']),
                   pretrained_weights="coco"
                   )

save_period = 20
save_ckpt_epoch_list = [i for i in range(100, MAX_EPOCHS, save_period)]
print(f"Save ckpt epoch list: {save_ckpt_epoch_list}")

num_updates = 3
start_update = 100
update_every_n_epochs = math.ceil((MAX_EPOCHS - start_update) / num_updates)
lr_updates = [100, 150, 200, 250]

#resume_path = 
train_params = {
    "resume": True,
    #"resume_path": resume_path,
    'silent_mode': False,
    # "MultiGPUMode": "DP",
    "average_best_models":True,
    "warmup_mode": "LinearEpochLRWarmup",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 3,
    "initial_lr": 1e-3,
    "lr_mode": "StepLRScheduler", # ExponentialLRScheduler
    "lr_updates": lr_updates,
    "optimizer": "AdamW",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "save_ckpt_epoch_list": save_ckpt_epoch_list,
    "lr_decay_factor": 0.2,
    "max_epochs": MAX_EPOCHS,
    "mixed_precision": True,
    "loss": PPYoloELoss(
        num_classes=len(dataset_params['classes']),
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.5,
            num_cls=len(dataset_params['classes']),
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7,
            ),
            include_classwise_ap = True,
            class_names = dataset_params['classes']
        ),
        DetectionMetrics_050_095(
            score_thres=0.5,
            top_k_predictions=300,
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7,
            ),
            include_classwise_ap = True,
            class_names = dataset_params['classes']
        ),
    ],
    "metric_to_watch": 'mAP@0.50:0.95'
}

trainer.train(
    model=model,
    training_params=train_params,
    train_loader=train_data,
    valid_loader=val_data
)

model_path = sorted(os.listdir(CHECKPOINT_DIR + '/' + EXPERIMENT_NAME))[-1]

best_model = models.get(
    "yolo_nas_s",
    num_classes=len(dataset_params['classes']),
    checkpoint_path= model_path + '/' + 'average_model.pth'
).to('cuda')

trainer.test(
    model=best_model,
    test_loader=test_data,
    test_metrics_list=[
        DetectionMetrics_050(
            score_thres=0.5,
            num_cls=len(dataset_params['classes']),
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7,
            ),
            include_classwise_ap = True,
            class_names = dataset_params['classes']
        ),
        DetectionMetrics_050_095(
            score_thres=0.5,
            top_k_predictions=300,
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7,
            ),
            include_classwise_ap = True,
            class_names = dataset_params['classes']
        ),
    ]
)

# inference on test set


ds = sv.DetectionDataset.from_yolo(
    images_directory_path=f"{LOCATION}/test/images",
    annotations_directory_path=f"{LOCATION}/test/labels",
    data_yaml_path=f"{LOCATION}/data.yaml",
    force_masks=False
)
CONFIDENCE_TRESHOLD = 0.4

predictions = {}

for image_name, image in ds.images.items():
    result = list(best_model.predict(image, conf=CONFIDENCE_TRESHOLD))[0]
    detections = sv.Detections(
        xyxy=result.prediction.bboxes_xyxy,
        confidence=result.prediction.confidence,
        class_id=result.prediction.labels.astype(int)
    )
    predictions[image_name] = detections
    

MAX_IMAGE_COUNT = 5

n = min(MAX_IMAGE_COUNT, len(ds.images))

keys = list(ds.images.keys())
keys = np.random.choice(keys, n, replace=False)

box_annotator = sv.BoxAnnotator()

images = []
titles = []

for key in keys:
    frame_with_annotations = box_annotator.annotate(
        scene=ds.images[key].copy(),
        detections=ds.annotations[key],
        skip_label=True
    )
    images.append(frame_with_annotations)
    titles.append('annotations')
    frame_with_predictions = box_annotator.annotate(
        scene=ds.images[key].copy(),
        detections=predictions[key],
        skip_label=True
    )
    images.append(frame_with_predictions)
    titles.append('predictions')

sv.plot_images_grid(images=images, titles=titles, grid_size=(n, 2), size=(2 * 4, n * 4))