# Software-Engineering_Make-Program

# Flask Image Analysis Program

This is a simple web application that allows you to upload an image, analyze it using YOLOv4, and display the analysis results on a web page.

## Requirements

To run this program, you need the following dependencies:

- Python 3
- Flask
- OpenCV
- Numpy
- YOLOv4 configuration file (yolov4.cfg)
- YOLOv4 weights file (yolov4.weights)
- YOLOv4 class names file (coco.names)

## Installation

1. Install the required Python packages using `pip`:

   ```bash
   pip install flask opencv-python-headless numpy

Download the YOLOv4 configuration, weights, and class names files, and place them in the same directory as the program.

Usage
1. Run the Flask application:

python your_app_name.py

3. Open your web browser and navigate to http://localhost:5000/.

4. You will see an image upload form. Select an image and click the "Start Analysis" button.

5. The program will analyze the image using YOLOv4 and display the result on a new web page.

6. You can click the "Analyze Another Image" link to upload and analyze more images.



Note

This program is a simple example and may require further security and error handling enhancements for production use.
Make sure to have the YOLOv4 configuration, weights, and class names files in the same directory as the program for it to work properly.
Enjoy using this image analysis program!
