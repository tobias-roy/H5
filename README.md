# Object Detection using Tensorflow 2, Transfer Learning and TensorflowJS in Angular19

[1. Case](#Case)

[2.1 Tensorflow 2 Installation Guide](#Tensorflow-2-Installation-Guide)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[2.11 Software Installation](#Softwar-Installation)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[2.12 Setting up the project](#Setting-up-the-project)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[2.13 Setting up PyCharm](#Setting-up-PyCharm)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[2.14 Setting up a model for Transfer Learning](#Setting-up-a-model-for-Transfer-Learning)

[2.2 Tensorflow 2 Local Training Guide](#Tensorflow-2-Local-Training-Guide)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[2.21 3rd-party Software installation](#3rd-party-Software-installation)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[2.22 Preparing the Dataset](#Preparing-the-Dataset)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[2.23 Model configuration](#Model-configuration)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[2.24 Training the model](#Training-the-model)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[2.25 Tensorboard analytics](#Tensorboard-analytics)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[2.26 Error handling](#Error-handling)

## TODO
[2.3 TensorflowJS Webcam Detection](#TensorflowJS-Webcam-Detection)


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
This part of the guide will set you up for training a model locally using transfer learning.

## 3rd-party Software installation
- Download and install the latest version of [LabelImg](https://github.com/HumanSignal/labelImg/releases)
  You can either do it from the link or in your Anaconda Environment usen this command
	```
  	pip install labelimg
	```
## Preparing the Dataset
In order to train a model you need a prepared dataset.
For the purpose of this guide i am trying to detect specific ships in still images and videofeeds.
You should have atleast 200 pictures, more is preferable.

When training a model the general rule of thumb for using a dataset is to do a 80 / 20 split, meaning that 80% of your images will be used for training and 20% will be used for validation testing.

If you don't want to create your own dataset [Kaggle](https://www.kaggle.com/datasets) is a great source for large datasets.

Gathering good data is a science of it's own and i will leave that up to you to figure out.

- When you have gathered your dataset create a new directory in your object_detection folder and name it *images*
- In *object_detection/images* create two subfolders one named *test* and one named *train*
- Paste 80% of your dataset into the *object_detection/images/train* directory and 20% into *object_detection/images/test*
- If your dataset is divided into more categories you can create subfolders in the *object_detection/images/train* directory for each category, but this is not a necessary step.

- Open LabelImg from your desktop or with the command ```labelImg.exe```in your Anaconda Environment
- Press the *Open Dir* button and open the *object_detection/images/train* directory
- Press the *Change Save Dir* button and change the save directory to *object_detection/images/train*
- Now it's time for the tedious part which is labeling your dataset.
	Use the following keybinds for a bit of a better experience
	**W** - Makes a new label bounding box to mark your object
	**CTRL + S** - Saves the current image configuration
	**A** - Navigates back 1 image
	**D** - Navigates forward 1 image.
	Remember to save the image each time before going on to the next one using **CTRL + S**

- Great! Now you are 80% of the way, go through the same process with the *object_detection/images/test* directory
  Remember to set the *Save Directory* to *object_detection/images/test*

- Once done with labelling your dataset create a file called *xml_to_csv.py* in your the object_detection directory
- Paste the following script into the *xml_to_csv.py* file

```
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    for folder in ['train', 'test']:
        image_path = os.path.join(os.getcwd(), ('images/' + folder))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(('images/'+folder+'_labels.csv'), index=None)
    print('Successfully converted xml to csv.')


main()
```

The script above will look for the *images/train* and *images/test* folders and convert the xml files to csv files.

- In your Anaconda Prompt execute the script by running ```python xml_to_csv.py```
- 
  You should now se a confirmation message in your terminal and two new files should have appeared in your images folder.

- Create a new file in the object_detection directory named *generate_tfrecord.py*
- Insert the following script into the *generate_tfrecord.py* file
  
```
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd

from tensorflow.python.framework.versions import VERSION
if VERSION >= "2.0.0a0":
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to images')
FLAGS = flags.FLAGS


''' 
*************************************************************************
Make sure to edit this method to match the labels you made with labelImg!
*************************************************************************
'''
def class_text_to_int(row_label):
    if row_label == 'Your_Label':
        return 1
    elif row_label == 'Your_Other_Label':
        return 2
    else:
        return None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
```

- Open the file and edit the *class_text_to_int* function to fit your dataset, add the names of the labels you used when labelling your images.
  You can expand it just as you wish.

- In your Anaconda Prompt from the *object_detection* directory run the following two commands to generate the tfrecords

python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

You should be prompted with a succesful message

## Model configuration

- In the explorer navigate to *object_detection/configs/tf2* and find the configuration file matching the name of the model you are using

- Copy the configuration file at paste it in the *object_detection* folder

- Open the configuration file with an IDE or text editor

- Change the *num_classes* parameter in the model to the amount of classes you have defined
  
- Change the *fine_tune_checkpoint* to the path of the ckpt-0.index file in your downloaded model folder
	i.e *object_detection/YOUR_MODEL_FOLDER/checkpoint/ckpt-0*
	Make sure '/' are '/' and not '\'
	Make sure to delete the .index fileextension from the path

- Change the *fine_tune_checkpoint_type* to *detection*
  
- Change the *batch_size* in the *train_config* to a higher number like 64 if you have a CUDA compatible GPU.
	Change it to a lower number if you do not as it will utilize your CPU.
	Recommended for CPU is to start as low as possible so try it out with 2 for starters.
	
- Change the *num_steps* in the *train_config*
	This is basicly the number of steps the model will use for training - try leaving it at default and monitor your training.
	Change it higher or lower and compare the outputs. Too high a number can cause overtraining.

- Change the *input_path* of the *train_input_reader* to the path of the *train.record*
	ie. *research/object_detection/train.record*

- Change the *input_path* of the *eval_input_reader* to the path of the test.record
	ie. *research/object_detection/test.record*
	
- Create a file in the *object_detection* directory named *labelmap.pbtxt*
	In that file you need to create a label map which is a structure of the labels you used for your images.
	**The id has to correspond to the return value of your *generate_tfrecord.py* file.**
	Change the label_X name to the name of your labels and expand as needed.
```
item {
	id: 1
	name: 'label_1'
}
item {
	id: 2
	name: 'label_2'
}
item {
	id: 3
	name: 'label_3'
}
```

- Change the *label_map_path* of the the *train_input_reader* to *research/object_detection/labelmap.pbtxt*
	
- Change the *label_map_path* of the *eval_input_reader* to *research/object_detection/labelmap.pbtxt*

- In the *object_detection* directory create a new directory and name it *training* add a subfolder with the name of the model you are using

## Training the model

- In your Anaconda Prompt, from the *object_detection* directory run the following command to start training
	```python model_main_tf2.py --pipeline_config_path=THE_PATH_OF_YOUR_CONFIG_FILE --model_dir=training --alsologtostderr```
	
	In my case the command looks like this:
	```python model_main_tf2.py --pipeline_config_path=faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.config --model_dir=training/faster_rcnn_resnet50_v1_640x640 --alsologtostderr```
	
	The *model_dir* is where training checkpoints will be stored once training begins.
	The aslologtostderr command will log standard errors.

- When the model is done training we can export and save the it for later use using this command
  	```python exporter_main_v2.py --trained_checkpoint_dir=training/faster_rcnn_resnet50_v1_640x640 --pipeline_config_path=faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.config --output_directory inference_graph```
	This will run the export script, use the checkpoints in the training folder, use the config that we made for the model and export it in the *object_detection/inferece_graph* folder.

## Tensorboard analytics

- When the model is running open another Anaconda Prompt, activate the environment you're using and 
navigate to the *object_detection* directory.
- Run the following command to enable TensorBoard
	```tensorboard --logdir=training\faster_rcnn_resnet50_v1_640x640\train```
	The command above points to the train folder containing tfevents files.

## Error handling

- The formatting of the labelmap.pbtxt file is very important. When you have the file open in an IDE such as PyCharm in the bottom hand right corner
make sure that the Line Sperator is CR and the file encoding is UTF-8.


# TensorflowJS Webcam Detection

- In order to use this model in Tensorflow.js we need to convert it.
Create a new Anaconda Environment with another version of python
```conda create -n tfjsconverter python=3.6.8```
Activate the environment
```activate tfjsconverter```

Install Tensorflowjs in your Anaconda environment
	```pip install tensorflowjs[wizard]```

- In your prompt navigate to the *object_detection* folder
 
- Run the command below to convert our model

```tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model inference_graph/saved_model/saved_model.pb inference_graph/saved_model/tfjsconvert```
	
 This will create a new directory and a graph model which can be used in TensorflowJS in the *inference_graph/saved_model* directory

- In order to use this model in TensorflowJS you have to upload the model.json and all the .bin files (Let's call them shards)
  tf.loadModel() a Javascript function that loads the model uses FETCH and has to have a valid link to point to.
  I upload mine to Github in a public repository and then point to the RAW path, but more on that in the next chapter.
  *https://raw.githubusercontent.com/tobias-roy/H5/refs/heads/MachineLearning/AngularTensorflowJS/model/ship-detector-resnet50/model.json*



# FAQ

#### The difference in Layer and Graph models
1. LayersModel can only be imported from tf.keras or keras HDF5 format model types. GraphModels can be imported from either the aforementioned model types, or TensorFlow SavedModels.

2. LayersModels support further training in JavaScript (through its fit() method). GraphModel supports only inference.

3. GraphModel usually gives you higher inference speed (10-20%) than LayersModel, due to its graph optimization, which is possible thanks to the inference-only support.

[Source Shanqing Cai](https://stackoverflow.com/a/59341012)


## Authors

- [@tobias-roy](https://github.com/tobias-roy)

  I created this guide with inspiration from Ben Greenfield [Link to his github](https://github.com/BenGreenfield825).
