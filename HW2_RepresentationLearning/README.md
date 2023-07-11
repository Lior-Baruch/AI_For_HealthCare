# Bone Fracture Classification using Representation Learning

This project aims to classify bone fractures using a two-step process: first, using self-supervised learning to learn a representation of the images, and then using that representation to classify the images.

## Data Preparation

The dataset used in this project consists of X-ray images of different body parts, each labeled as either "positive" or "negative" for a fracture. The images are organized into directories by body part and patient. The `load_data` function is used to read in the images and store their file paths, labels, and body parts into a pandas DataFrame. The labels are also mapped from strings ("positive" and "negative") to integers (1 and 0).

```python
def load_data(data_dir):
    ...
```

The data is then split into training and validation sets using scikit-learn's `train_test_split` function. 

```python
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
```

## Data Augmentation

Data augmentation is performed using torchvision's transforms module. The images are resized to 224x224 pixels, converted to tensors, and normalized to have a mean of 0.5 and a standard deviation of 0.5.

```python
train_transforms = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

The datasets are then prepared using a custom `ImageDataset` class, which loads and preprocesses the images on-the-fly when they are requested. 

```python
class ImageDataset(Dataset):
    ...
```

DataLoaders are then created for the training, validation, and test sets.

```python
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
```

## Baseline Model

The baseline model is a simple convolutional neural network (CNN) with two convolutional layers and two fully connected layers. 

```python
class SimpleCNN(nn.Module):
    ...
```

The model is trained using binary cross-entropy loss and the Adam optimizer. The training loop consists of the standard forward pass, loss computation, backward pass, and parameter update steps. The training and validation loss and accuracy are recorded for each epoch.

```python
for epoch in tqdm(range(n_epochs)):
    ...
```

The trained model and the training history are saved to disk.

```python
torch.save(base_model.state_dict(), 'base_model.pth')
torch.save({'train_loss_history': train_loss_history,
            'valid_loss_history': valid_loss_history,
            'train_acc_history': train_acc_history,
            'valid_acc_history': valid_acc_history}, 'base_model_history.pth')
```

The training and validation accuracy and loss are then plotted over the epochs.

```python
plt.figure(figsize=(12, 6))
...
plt.show()
```

## BYOL Model

For the self-supervised learning step, the BYOL (Bootstrap Your Own Latent) method is used. This involves training a model to predict its own output when given a different view of the same input.

First, additional data augmentations are performed to create the new views of the images. These include random vertical flipping and rotation.

```python
train_new_view_transforms = transforms.Compose([
    transforms.Resize(size),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

The BYOL model is then defined. It consists of a backbone (a ResNet model), a projection head, and a prediction head. The backbone is pre-trained and its parameters are frozen.

```python
class BYOL(nn.Module):
    ...
```

The model is trained using a custom loss function that encourages the model's predictions for the two views of the same image to be similar. The target model is updated with a moving average of the model's parameters.

```python
for epoch in tqdm(range(epochs)):
    ...
```

The trained model is then saved to disk.

```python
torch.save(byol_model.state_dict(), 'byol_model.pth')
```

## Classifier

Finally, a classifier is built on top of the BYOL model. The classifier uses the learned representations from the BYOL model to perform the final classification task.

```python
class BYOLClassifier(nn.Module):
    ...
```

The classifier is trained using cross-entropy loss and the Adam optimizer. The training loop is similar to the one used for the baseline model.

```python
for epoch in tqdm(range(num_epochs)):
    ...
```

The trained classifier and the training history are saved to disk.

```python
torch.save(classification_byol_model.state_dict(), 'classification_byol_model.pth')
torch.save({'train_loss_history_byol': train_loss_history_byol,
            'train_acc_history_byol': train_acc_history_byol,
            'val_loss_history_byol': val_loss_history_byol,
            'val_acc_history_byol': val_acc_history_byol}, 'history_byol.pth')
```

The training and validation accuracy and loss are then plotted over the epochs.

```python
plt.figure(figsize=(10, 5))
...
plt.show()
```

To use this code, you will need to install the necessary packages and provide the appropriate dataset. You can adjust the hyperparameters and the architecture of the models to better suit your specific task.
