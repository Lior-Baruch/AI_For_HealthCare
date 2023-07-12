# README: Bone Fracture Classification using Self-Supervised Learning

## Project Overview

This project demonstrates the application of self-supervised learning, specifically the Bootstrap Your Own Latent (BYOL) method, for classifying bone fractures. It utilizes a two-step process: first, the model learns a representation of X-ray images of bones, and then uses that representation to classify the images into 'normal' and 'abnormal' categories. The dataset used consists of X-ray images of three types of bones: the shoulder, hand, and elbow.

## Data Preparation

The X-ray images, each labeled as 'normal' or 'abnormal', are organized into directories by body part and patient. The `load_data` function reads the images and stores their file paths, labels, and body parts into a pandas DataFrame. The labels are also mapped from strings to integers.

The dataset is split into training and validation sets using scikit-learn's `train_test_split` function. Data augmentation techniques, including resizing, converting to tensors, and normalization, are performed on the images using torchvision's transforms module.

The images are loaded and preprocessed on-the-fly when they are requested using a custom `ImageDataset` class, and DataLoaders are created for the training, validation, and test sets.

## Baseline Model

The baseline model is a simple convolutional neural network (CNN) with two convolutional layers and two fully connected layers. The model is trained using binary cross-entropy loss and the Adam optimizer. The training and validation loss and accuracy are recorded for each epoch.

## Advanced Approach - BYOL

The advanced approach involves the BYOL method, which trains a model to predict its own output when given a different view of the same input. The BYOL model consists of a backbone (a ResNet model), a projection head, and a prediction head. The backbone is pre-trained and its parameters are frozen during training.

Additional data augmentations are performed to create new views of the images, including random vertical flipping and rotation. The model is trained using a custom loss function that encourages the model's predictions for the two views of the same image to be similar. The target model is updated with a moving average of the model's parameters.

## Classifier

Finally, a classifier is built on top of the BYOL model. The classifier uses the learned representations from the BYOL model to perform the final classification task. The classifier is trained using cross-entropy loss and the Adam optimizer.

## Results

After training, the performance of the models is evaluated on a test set. The baseline model achieved a test accuracy of 53.16%, while the classifier model reached a test accuracy of 67.81%. This represents an improvement of approximately 14.65 percentage points.

## Dependencies

This code relies on the following libraries:
- numpy
- pandas
- scikit-learn
- torch
- torchvision
- tqdm

## Usage

To use this code, you will need to install the necessary packages and provide the appropriate dataset. You can adjust the hyperparameters and the architecture of the models to better suit your specific task. Please refer to the Jupyter notebook for detailed implementation and instructions. 

## Future Work

Training the BYOL model was computationally more expensive due to its complex architecture. However, the improvement in accuracy indicates that the additional computational effort can be justified. Further improvements could potentially be made by using more sophisticated data augmentation techniques, tuning the hyperparameters, and fine-tuning the BYOL model on a downstream task using labeled data.
