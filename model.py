from ultralytics_ import YOLO
from predictor import DetectionPredictor
import torch

class MODEL:
    """
    A class representing a model for court vision.

    Args:
        model_path (str): The path to the model file.
        device (str): The device to run the model on.
        imgsz (int): The size of the input images.

    Attributes:
        model: The YOLO model.
        processor: The detection predictor.

    Methods:
        predict: Predicts the court vision for a given image or a list of images.
    """

    def __init__(self, model_path, device):
        self.model = YOLO(model_path).model.half()
        self.model.to(device)
        self.model.eval()
        self.imgsz = self.model.args['imgsz']
        self.processor = DetectionPredictor(device=device, imgsz=self.imgsz, model=self.model)
        

    def predict(self, imgs):
        """
        Predicts the court vision for a given image or a list of images.

        Args:
            imgs: The input image(s) to predict court vision for.

        Returns:
            The predicted court vision result(s).
        """
        if not isinstance(imgs, list):
            imgs = [imgs]
        with torch.no_grad():
            preprocessed_tensor = self.processor.preprocess(imgs)
            raw_output = self.model(preprocessed_tensor)[0]
            result = self.processor.postprocess(raw_output, preprocessed_tensor, imgs)
        return result