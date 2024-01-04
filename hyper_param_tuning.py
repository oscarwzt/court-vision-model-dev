import ray
from ray import tune
from ray.tune.schedulers import HyperBandScheduler
import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from utils import * 
from training_utils import *

data_full_path = "/scratch/zw2688/Court_Vision_Model_Dev/data/classification_dataset_groupby_env_split"      
 
def train_resnet(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)
    train_loader, valid_loader, _ = get_data_loaders(config["batch_size"], config["img_size"],  data_full_path, config["normalize"])
    
    if config["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config['momentum'], nesterov=config["nestrov"])
    elif config["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=config["lr"])

    criterion = nn.BCEWithLogitsLoss()
    while True:
        train_epoch(model, optimizer, criterion, train_loader, device)
        _, test_accuracy = test(model, criterion, valid_loader, device)
        ray.train.report(metrics = {"accuracy": test_accuracy})
 
batch_sizes = [64, 128, 256]
img_sizes = [224, 112, 96]
optimizers = ["sgd", "rmsprop", "adam"]

# hyperband
hyperband_scheduler = HyperBandScheduler(
    time_attr='training_iteration',
    metric='accuracy',
    mode='max',
    max_t=81,
    reduction_factor=3)

hyperband_analysis = tune.run(
    train_resnet,
    name="tuning_cls_resnet18_augmented",
    stop={
        "accuracy": 0.92,
        "training_iteration": 100
    },
    resources_per_trial={
        "gpu": 0.5,
        "cpu": 2
    },
    config={
        "lr": tune.loguniform(8e-5, 5e-3),
        "batch_size": tune.grid_search(batch_sizes),
        "optimizer": tune.grid_search(optimizers),
        "img_size": tune.grid_search([224, 112, 96]),
        "nestrov": tune.grid_search([True, False]),
        "momentum": tune.uniform(0.6, 0.95),
        "normalize": tune.grid_search([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
            None]),
    },
    scheduler=hyperband_scheduler,
    storage_path = "/scratch/zw2688/Court_Vision_Model_Dev/tune_results",
    )

