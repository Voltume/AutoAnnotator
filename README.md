# Project Name: Object Detection Annotation Generator

## Overview

This project, conducted at ENSAM-Meknes, focuses on simplifying the process of generating bounding boxes for objects in images. The primary objective is to provide students with a fast and reliable tool to create annotations for object detection projects. The current implementation utilizes the GroundingDINO model to generate bounding boxes for a given set of classes. The generated annotations are saved in the Pascal VOC format, enabling seamless integration with various object detection frameworks.

## Features

- **GroundingDINO Model Integration**: The project leverages the power of the GroundingDINO model to efficiently generate bounding boxes for specified classes within images.

- **Pascal VOC Annotation Format**: Annotations are saved in the Pascal VOC format, a widely used standard for object detection tasks. This format includes information such as object bounding boxes, class labels, and image metadata.

## Usage

1. **Installation**: Clone the repository and install the required dependencies.

    ```bash
    git clone [https://github.com/your-username/annotation-generator.git](https://github.com/Voltume/AutoAnnotator.git)
    cd AutoAnnotator
    pip install -r requirements.txt
    chmod +x dependencies.sh
    ./dependencies.sh
    ```

2. **Add data path and classes**: in the application.py file change the variables `DATA_PATH` and `CLASSES`

3. **Run the python code**: Use the following command to generate masks

    ```bash
    python application.py
    ```

    This command will use the GroundingDINO model to generate bounding boxes for the specified classes in the given images.

4. **Output**: The annotations will be saved in the Pascal VOC format in the `annotations` directory.

## Next Steps

The following enhancements are planned for future releases:

- **Support for Multiple Annotation Formats**: Extend the tool to generate annotations in additional formats such as YOLO, COCO, etc., to accommodate a wider range of object detection frameworks.

- **Segmentation Mask Generation**: Implement functionality to generate segmentation masks for objects in images, providing more detailed annotations for advanced computer vision tasks.

## Contributing

Contributions to the project are welcome! If you have ideas for improvements or encounter any issues, feel free to open an [issue](https://github.com/Voltume/AutoAnnotator/issues) or submit a [pull request](https://github.com/Voltume/AutoAnnotator/pulls).

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute the code for academic and non-commercial purposes.
