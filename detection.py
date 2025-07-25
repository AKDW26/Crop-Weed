import cv2
import numpy as np
import supervision as sv
from roboflow import Roboflow
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class DetectionResult:
    image: np.ndarray
    detections: List[Dict]
    class_counts: Dict[str, int]

class CropWeedDetector:
    def __init__(self, api_key: str):
        self.rf = Roboflow(api_key=api_key)
        self.project = self.rf.workspace().project("agriweed")
        self.model = self.project.version(1).model
        
        # Initialize annotators
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def _format_detections(self, predictions: List[dict], image_shape: Tuple[int, int]) -> sv.Detections:
        if not predictions:
            return sv.Detections.empty()

        boxes = []
        confidence = []
        class_ids = []

        for pred in predictions:
            x1 = pred['x'] - pred['width'] / 2
            y1 = pred['y'] - pred['height'] / 2
            x2 = pred['x'] + pred['width'] / 2
            y2 = pred['y'] + pred['height'] / 2
            
            boxes.append([x1, y1, x2, y2])
            confidence.append(pred['confidence'])
            class_ids.append(pred['class'])

        return sv.Detections(
            xyxy=np.array(boxes),
            confidence=np.array(confidence),
            class_id=np.array([hash(class_id) % 32 for class_id in class_ids])
        )

    def _count_classes(self, predictions: List[dict]) -> Dict[str, int]:
        class_counts = {}
        
        for pred in predictions:
            class_name = pred['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
        return class_counts

    def process_image(self, image: np.ndarray, confidence: float = 40, overlap: float = 30) -> DetectionResult:
        result = self.model.predict(image, confidence=confidence/100.0, overlap=overlap/100.0).json()
        
        predictions = result['predictions']
        
        detections = self._format_detections(predictions, image.shape[:2])
        
        annotated_image = image.copy()
        
        annotated_image = self.box_annotator.annotate(scene=annotated_image, detections=detections)
        
        class_names = [pred['class'] for pred in predictions]
        
        annotated_image = self.label_annotator.annotate(scene=annotated_image, detections=detections, labels=class_names)
        
        class_counts = self._count_classes(predictions)
        
        return DetectionResult(image=annotated_image, detections=predictions, class_counts=class_counts)

    def get_detection_stats(self, result: DetectionResult) -> Dict:
        stats = {
            "total_detections": len(result.detections),
            "class_distribution": result.class_counts,
            "detection_confidence": {
                "mean": np.mean([d['confidence'] for d in result.detections]) if result.detections else 0,
                "min": np.min([d['confidence'] for d in result.detections]) if result.detections else 0,
                "max": np.max([d['confidence'] for d in result.detections]) if result.detections else 0,
            }
        }
        
        return statsgkhasgfdiadgfi
