# Object Detection using Tensorflow 2 and TensorflowJS in Angular19

[1. Case](#Case)

[2.1 Tensorflow 2 Installation Guide](#Tensorflow-2-Installation-Guide)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[2.11 Software Installation](#Softwar-Installation)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[2.12 Setting up the project](#Setting-up-the-project)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[2.13 Setting up PyCharm](#Setting-up-PyCharm)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[2.14 Setting up a model for Transfer Learning](#Setting-up-a-model-for-Transfer-Learning)

[2.2 Tensorflow 2 Local Training Guide](#Tensorflow-2-Local-Training-Guide)

[2.3 TensorflowJS Webcam Detection Guide](#Tensorflow)

[2.4 TensorflowJS Still Image Detection Guide](#Tensorflow)

[3. Dataset](#Dataset)

[4. Solution](#Solution)

[5. Metrics](#Metrics)

[6. TLDR](#TLDR)

# Case
Sailing ships around the world you might be exposed to all kinds of danger, one in particulair is the threat of pirates. 
Pirates are not exclusive to the horn of Africa, there is an ecually high risk in The Strait of Malacca and Gulf of Guinea.
The M.O. of pirates is typicaly similair in regards to transportation, they tend to use small fast speed-boats. Hard to hit, easy to sail.

To help ships navigate in the waters we can implement an Early Warning System (EWS). This will help ships detect and identify possible threads.
At the top of the ships a camera will be mounted that contionuously will rotate in a 360 degree rotation to constantly survey the waters.
The videofeed will then be used by our object detection model to try and identify pirates.


# Tensorflow 2 Installation Guide
This guide take any skill level into account, you can follow this as a complete beginner in ML with very basic computer skills.
If you don't want to follow the guide and learn along there will be a TLDR as short as possible.
It's very important that you install the specific version numbers in this guide to avoid package conflicts.

## Software installation

- Clone the tensorflow [models repository](https://github.com/tensorflow/models) into a project folder you want to continue from.
  ```
  gh repo clone tensorflow/models
  ```

- Install [Anaconda](https://www.anaconda.com/download/success) for managing the python environment
	
- Install [PyCharm](https://www.jetbrains.com/pycharm/) or another IDE with Python compatability


## Setting up the project
- Open an Anaconda Prompt
- Create a new environment ```conda create -n *YOUR-PROJECT-NAME* python=3.9```
- Activate the project: ```conda activate projectName```
- Download protobuf as a package in your environment ```conda install protobuf=3.20.1```
- Copy the directory path to your tensorflow/models/research directory (models should be the cloned [models repository](https://github.com/tensorflow/models))
- Navigate to the directory in your Anaconda Prompt ```cd C:\PATH\OF\YOUR\FOLDER\models\research```
- Create a file named use_protobuf.py in *C:\PATH\OF\YOUR\FOLDER\models\research*
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

- Run the script in your Anaconda Prompt with the command ```python use_protobuf.py object_detection/protos protoc```
This will activate all the protobuf necessary stuff

- In your explorer navigate to *models/research/object_detection/packages/tf2*
- Copy the *setup.py* file
- Navigate back to *models/research*
- Paste the *setup.py* file

- In your Anaconda prompt run ```python -m pip install .```
That will run the setup

- In your Anaconda prompt run ```conda install numpy=1.23.4```
  This will install Numpy

- In your Anaconda prompt run ```python object_detection/builders/model_builder_tf2_test.py```
  This is a Tensorflow Test to see if everything is installed correctly so far.

## Setting up PyCharm
This step will vary depending on what IDE you are using but for PyCharm follow along.
- In your explorer navigate to your *models* folder
- Press SHIFT + Right Click and open the folder *As a project for PyCharm*
- Click the interpeter in the bottom right corner (The second field from the right)
- Press *Add New Interpreter* and then *Add Local Intepreter*
- Press *Select Existing* then in the Type dropdown choose *Conda* then in the Environment dropdown choose the name of the environment you created earlier

## Setting up a model for Transfer Learning
- Create a new directory in the *models/research/object_detection* folder and name it *outputs*
- Create a new file in the *models/research/object_detection* folder and name it *model_downloader.py*
- Create a new file in the *models/research/object_detection* folder and name it *detect_from_image.py*

- Insert the following script into the *model_downloader.py* file
```
import wget
model_link = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz"
wget.download(model_link)
import tarfile
tar = tarfile.open('faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz')
tar.extractall('.')
tar.close()
```
In case you get a red underline on wget you need to install it in your environment.
- In your Anaconda prompt run ```pip install wget=3.2```

The code above is pointing to the faster_rcnn_resnet50_v1_640x640_coco17_tpu.8 model. You can choose whatever Tensorflow 2 Object Detection model you would like to use.
The [Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) containts a list of all models we can use.
- Right click the name of the model you want and press *Copy Link*
- Replace the model_link with the link you copied.
- Make sure the tarfile.open has the correct .tar.gz filename correspoding to the name of the model you downloaded and want to use.

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

- In your Anaconda Prompt make sure you're in the *models/research/object_detection/* folder
- Run the model downloader script ```python model_downloader.py```

- Make sure the following command points to the model you just downloaded and then run
- ```python detect_from_image.py -m faster_rcnn_resnet50_v1_640x640_coco17_tpu-8\saved_model -l data\mscoco_label_map.pbtxt -i test_images```

What this command does is that it runs the detect_from_image python script with the model that we just downloaded.
	It will use the pre-defined object labels in the *mscoco_label_map.pbtxt* which is a collection that most pretrained object detection models can use.
	For more information of the [Common Objects in Context](https://cocodataset.org/#home) dataset visit the webpage.
	
In your outputs folder you should now see a couple of sample images that have object detection added to them.


# Tensorflow 2 Local Training Guide




## FAQ

#### Is ParkingProject a fully functioning console application?

Absolutely yes.

#### Can i use it for anything?

Absolutely not.


## Authors

- [@tobias-roy](https://github.com/tobias-roy)
