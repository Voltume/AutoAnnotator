from AutoAnnotator import AutoAnnotator

DATA_PATH = None
TARGE_DIR = "./annotations"
CLASSES = ['car', 'dog', 'person', 'nose', 'chair', 'shoe', 'ear', 'coffee', 'backpack', 'cap']

annotator = AutoAnnotator(classes=CLASSES)
images, annotations = annotator.get_boxes()
annotator.plot_boxes(images, annotations)
annotator.to_xml()