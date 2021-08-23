#### Acknowledgments

The code in this repository has been copied from the original repo : 
https://github.com/sniper0110/YOLOv3 with minor modifications in the code.
I added more instruction on how to run the code and will soon add instruction 
on how to run it on AWS SageMaker. I shall also highly recomend the Udemy class 
associated with it https://www.udemy.com/course/deep-learning-for-object-detection-using-tensorflow-2.

# YOLO-v3 model tutorial

## Environement setup
`python3 -m venv .env`
`source .env/bin/activate`
`python3 -m pip install -r docs/requirements.txt`

In this file we will explain how to use this module in order to train easily new YOLO-v3 model on new datasets.
The dataset we will use for this tutorial can be found at the following url: https://public.roboflow.com/object-detection/mask-wearing
I should recomande to download the TensorFlow Object Detection CSV fomrat.

## Prepairing the dataset

### Annotation files

The `(train|test)_mask_annotations.txt` files located in the folder `data/dataset`
contains the annotation of your datasets.
Each line corespond to one image in your training or testing set. The line must be formed as follow:
`<global_path_to_the_image> <x_min_box_1>,<y_min_box_1>,<x_max_box_1>,<y_max_box_1>,<class_index_box_1> <x_min_box_2>...`

### Class names

The class names of the objects from your dataset must be indicated in the file
`data/classes/<project_name>.names`. The file must have the following form:
```
class_name_0
class_name_1
class_name_2
...
```

### Create the files from the csv raw data

```
python prepare_data.py \
--path_to_images ${PWD}/data/csv_raw_data/train \
--path_to_csv_annotations ${PWD}/data/csv_raw_data/train/_annotations.csv \
--path_to_save_output ${PWD}/data/images/train

python prepare_data.py \
--path_to_images ${PWD}/data/csv_raw_data/test \
--path_to_csv_annotations ${PWD}/data/csv_raw_data/test/_annotations.csv \
--path_to_save_output ${PWD}/data/images/test

python prepare_data.py \
--path_to_images ${PWD}/data/csv_raw_data/valid \
--path_to_csv_annotations ${PWD}/data/csv_raw_data/valid/_annotations.csv \
--path_to_save_output ${PWD}/data/images/valid

# We will use the validation in the training as we don't have too much data 
# Better not to do so if you have a lot of annotated data
mv data/images/valid/*.jpg data/images/train

cat data/images/valid/annotation.txt | sed -e 's/valid/train/g' >> data/images/train/annotation.txt

mv data/images/train/annotation.txt data/dataset/train_mask_annotation.txt
mv data/images/test/annotation.txt data/dataset/test_mask_annotation.txt
```

## Configuration of the training

The cofiguration file is location in `core/config.py`.

The important paramter that you should set depending the phase of your project 
and the capacity of your machine are the following:
- __C.TRAIN.BATCH_SIZE (The batch size)
- __C.TRAIN.EPOCHS (The number of epochs)
- __C.TRAIN.DATA_AUG (Weather you want data augmentation, advised to set at True)
- __C.TRAIN.LR_INIT (The learning rate at the begining of the training)
- __C.TRAIN.LR_END (The learning rate at the end of the training)

## Launch the training

In order to laucnh the training local, you simply need to run the following command:
`python train.py`

In order to follow the training process, you can use tensorboard with the event
file created during the training with the following commande:
`tensorboard --logdir './data/log'`


## Evaluate the model

To make the prediction on the test set you can simply use the following python script:

`python test.py`

In order to compute the mean average precision you can use the built python 
script for it with the folowing command:

`python mAP/main.py`

This will create a lot of different visualisation on the model performances that
you will be able to find in the folder `results`.

## Deploying the model in production

When you deploy your model in production, you should not use the check point format.
This is used to save the model at a certain state of it's training. Instead you 
should use the [SavedModel](https://www.tensorflow.org/guide/saved_model) format.
In order to convert your checkpoint model into a SavedModel format model you 
should use the following python script:

`python ckpt_to_savedModel.py`

This will save your model in the folder `SavedModel/YOLOv3_model`


