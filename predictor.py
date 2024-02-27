from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
import torch
import numpy as np

class BasePredictor:
    def __init__(self, device, imgsz, model):
        self.device = device
        self.imgsz = imgsz
        self.model = model

    def preprocess(self, im):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() # if self.model.fp16 else im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

    def pre_transform(self, im):
        """
        Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Returns:
            (list): A list of transformed images.
        """
        same_shapes = all(x.shape == im[0].shape for x in im)
        letterbox = LetterBox(self.imgsz, auto=same_shapes, stride=32)
        return [letterbox(image=x) for x in im]



class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model='yolov8n.pt', source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """
    def __init__(self, conf=0.25, iou=0.45, agnostic_nms=False, max_det=1000, classes=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = args
        self.conf = conf
        self.iou = iou
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        self.classes = classes

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds,
            self.conf,
            self.iou,
            agnostic=self.agnostic_nms,
            max_det=self.max_det,
            classes=self.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        orig_img_shape = orig_imgs[0].shape
        results = []
        

        for i, pred in enumerate(preds):
            # orig_img = orig_imgs[i]
            # results.append(Results(orig_img, path=None, names=self.model.names, boxes=pred))
            xyxy = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img_shape).int()
            cls = pred[:, -1].int()
            prob = pred[:, 4]
            
            result = {
                "boxes": xyxy.cpu().numpy(),
                "scores": prob.cpu().numpy(),
                "labels": cls.cpu().numpy(),
            }
            results.append(result)
        return results
