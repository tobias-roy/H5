# 
# Object Detection using ML.NET  

[1. Problem](#Problem)

[2. Solution](#Solution)

[3. Prerequisites](#Prerequisites)

[3.1 Tensorflow 2 Guide](#Tensorflow)

[4. Dataset](#Dataset)

[5. Solution](#Solution)

[6. Metrics](#Metrics)


## Problem
Sailing ships around the world you might be exposed to all kinds of danger, one in particulair is the threat of pirates. 
Pirates are not exclusive to the horn of Africa, there is an ecually high risk in The Strait of Malacca and Gulf of Guinea.
The M.O. of pirates is typicaly similair in regards to transportation, they tend to use small fast speed-boats. Hard to hit, easy to sail.

## Solution
In order to help ships 

## Prerequisites

## Tensorflow
Steps to get going with Tensorflow locally using Python 3.9:

Prerequisites:
- Clone the tensorflow models repository into a project folder you want to continue from
	https://github.com/tensorflow/models

- Download Protobuf for your platform
	https://github.com/protocolbuffers/protobuf/releases
	In my case its as of writing this protoc-21.10-win64
	
	Once downloaded store it somewhere you wont delete it (could be your Programs > Python folder
	Goto: System Properties tab -> Advanced tab -> Click Environment Variables
	Press "Path" in User variables -> Press Edit
	Press "New" and add the path of your folder containing the protoc.exe file.
	
- Install Anaconda for hosting the python environment
	https://www.anaconda.com/download/success
	
- Install PyCharm
	https://www.jetbrains.com/pycharm/

- Open an Anaconda prompt
	Create a new project with the command: 'conda create -n projectName python=3.9'
	Activate the project: 'conda activate projectName'
	Download protobuf as a package in your python environment: 'conda install protobuf=3.20.1'
	Copy the directory path containing your models folder (tensorflow models from previous step)
	In your Conda CMD navigate to the folder i.e 'cd C:\Users\tobia\source\repos\H5\PythonProject'
	Then 'cd models/research'
	
- In your explorer navigate to the models > research folder in the tensorflow/models cloned repo.
- Create a file named use_protobuf.py
- Insert the following script to the .py file
  
```
import os
import sys
args = sys.argv
directory = args[1]
protoc_path = args[2]
for file in os.listdir(directory):
	if file.endswith(".proto"):
		os.system(protoc_path+" "+directory+"/"+file+" --python_out=.")
```

- Run the script 'python use_protobuf.py object_detection/protos protoc'
That will input the directory and use the command protoc to activate all the protobuf needed tools

- In your explorer navigate to models > research > object_detection > packages > tf2
- Copy the setup.py file
- Navigate back to models > research
- Paste the setup.py file

- In Conda run 'python -m pip install .'
That will look for a setup.py file to install

- Install numpy 'conda install numpy=1.23.4'

- Test if everything is installed correctly 'python object_detection/builders/model_builder_tf2_test.py'

If everything runs, congratulations! You now have Tensorflow2 running.

The following is only relevant for PyCharm:
- Open your projects root folder as a project for PyCharm
- Click the interpeter in the bottom right corner -> Add new -> Select Local
- Press select excisting -> Choose Conda -> Choose the name of your environment with TensorFlow2

- Create a new directory in the object_detection folder and name it 'outputs'
- Create a new file in the object_detection folder and name it 'model_downloader.py'
- Create a new file in the object_detection folder and name it 'detect_from_image.py'

- Insert the following snippet into the mode_downloader.py file
  
```
import wget
model_link = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz"
wget.download(model_link)
import tarfile
tar = tarfile.open('faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz')
tar.extractall('.')
tar.close()
```

- Insert the following snippet into the detect_from_image.py file

```
import numpy as np
import argparse
import os
import tensorflow as tf
from PIL import Image
from io import BytesIO
import glob
import matplotlib.pyplot as plt

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: a file path (this can be local or on colossus)

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(model, image):
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def run_inference(model, category_index, image_path):
    if os.path.isdir(image_path):
        image_paths = []
        for file_extension in ('*.png', '*jpg'):
            image_paths.extend(glob.glob(os.path.join(image_path, file_extension)))

        """add iterator here"""
        i = 0
        for i_path in image_paths:
            image_np = load_image_into_numpy_array(i_path)
            # Actual detection.
            output_dict = run_inference_for_single_image(model, image_np)
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks_reframed', None),
                use_normalized_coordinates=True,
                line_thickness=8)
            """The existing plt lines do not work on local pc as they are not setup for GUI
                Use plt.savefig() to save the results instead and view them in a folder"""
            plt.imshow(image_np)
            # plt.show()
            plt.savefig("outputs/detection_output{}.png".format(i))  # make sure to make an outputs folder
            i = i + 1
    # else:
    #     image_np = load_image_into_numpy_array(image_path)
    #     # Actual detection.
    #     output_dict = run_inference_for_single_image(model, image_np)
    #     # Visualization of the results of a detection.
    #     vis_util.visualize_boxes_and_labels_on_image_array(
    #         image_np,
    #         output_dict['detection_boxes'],
    #         output_dict['detection_classes'],
    #         output_dict['detection_scores'],
    #         category_index,
    #         instance_masks=output_dict.get('detection_masks_reframed', None),
    #         use_normalized_coordinates=True,
    #         line_thickness=8)
    #     plt.imshow(image_np)
    #     plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects inside webcam videostream')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model Path')
    parser.add_argument('-l', '--labelmap', type=str, required=True, help='Path to Labelmap')
    parser.add_argument('-i', '--image_path', type=str, required=True, help='Path to image (or folder)')
    args = parser.parse_args()

    detection_model = load_model(args.model)
    category_index = label_map_util.create_category_index_from_labelmap(args.labelmap, use_display_name=True)

    run_inference(detection_model, category_index, args.image_path)
```


- In the above codesnippet we download and extract the resnet50 model for object detection. This can be changed to any model you wish to use from here:
	https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

- Run the model downloader script in your terminal (make sure you're in the same directory as the script)
	'python model_downloader.py'


- You should now be able to run the previously downloaded model for object detection, i used the Anaconda Prompt we set up earlier
- Navigate to models > research > object detection
- Run 'python detect_from_image.py -m faster_rcnn_resnet50_v1_640x640_coco17_tpu-8\saved_model -l data\mscoco_label_map.pbtxt -i test_images'
	What this command does is that it runs the detect_from_image python script with the resnet model that we downloaded earlier, if you have 
	another model make sure to addapt the model name in the commandline. It then tells the prompt to use the specific label map and then run it on the folder
	test_images. This will create an output for us.
	
- In your outputs folder you should now see images that have object detection added to them.




## FAQ

#### Is ParkingProject a fully functioning console application?

Absolutely yes.

#### Can i use it for anything?

Absolutely not.


## Authors

- [@tobias-roy](https://github.com/tobias-roy)
