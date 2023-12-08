import os
import torch
import cv2
import supervision as sv
from typing import List
from tqdm.notebook import tqdm
from groundingdino.util.inference import Model

GROUNDING_DINO_CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "weights/groundingdino_swint_ogc.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

def enhance_class_name(class_names: List[str]) -> List[str]:
  return [
      f"all {class_name}s"
      for class_name
      in class_names
  ]

class AutoAnnotator:

  def __init__(self, classes, images_path="./data"):
    self.images_path = images_path
    self.classes = classes
    self.images = {}
    self.annotations = {}
    self.plot_images = []
    self.plot_titles = []

  def get_boxes(self, transform=enhance_class_name, box_treshold=0.35, text_treshold=0.25, images_extensions = ['jpg', 'jpeg', 'png']):

    image_paths = sv.list_files_with_extensions(
        directory=self.images_path, 
        extensions=images_extensions)

    for image_path in tqdm(image_paths):
      image_name = image_path.name
      image_path = str(image_path)
      image = cv2.imread(image_path)

      detections = model.predict_with_classes(
          image=image,
          classes=transform(self.classes),
          box_threshold=box_treshold,
          text_threshold=text_treshold
      )
      detections = detections[detections.class_id != None]
      self.images[image_name] = image
      self.annotations[image_name] = detections
    return self.images, self.annotations
    
  def plot_boxes(self, images, annotations):
    box_annotator = sv.BoxAnnotator()

    for image_name, detections in annotations.items():
        image = images[image_name]
        self.plot_images.append(image)
        self.plot_titles.append(image_name)

        labels = [
            f"{self.classes[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _ 
            in detections]

        annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
        self.plot_images.append(annotated_image)
        title = " ".join(set([
            self.classes[class_id]
            for class_id
            in detections.class_id
        ]))
        self.plot_titles.append(title)

    sv.plot_images_grid(
        images=self.plot_images,
        titles=self.plot_titles,
        grid_size=(len(annotations), 2),
        size=(2 * 4, len(annotations) * 4)
    )

  def to_xml(self, target_dir = "./annotations", min_image_area_percentage = 0.002, max_image_area_percentage = 0.80, approximation_percentage = 0.75):
    sv.Dataset(
        classes=self.classes,
        images=self.images,
        annotations=self.annotations
    ).as_pascal_voc(
        annotations_directory_path=target_dir,
        min_image_area_percentage=min_image_area_percentage,
        max_image_area_percentage=max_image_area_percentage,
        approximation_percentage=approximation_percentage
    )