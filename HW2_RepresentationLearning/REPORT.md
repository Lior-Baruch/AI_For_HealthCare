# Report: Representation Learning for Bone Fracture Detection

## Introduction

This project focused on the classification of X-ray images of bones into 'normal' and 'abnormal' categories. The MURA dataset, containing 14,863 musculoskeletal radiographs from 12,173 patients (totalling 40,561 multi-view radiographic images), was utilized. This study specifically concentrated on three types of bones: the shoulder, hand, and elbow.

## Data Exploration

Initial data exploration revealed that the dataset is imbalanced, with more 'normal' cases than 'abnormal' ones. Furthermore, the distribution of bone types (shoulder, hand, elbow) was analyzed. 

- (plots showing the distribution of classes and the distribution of bone types)
- ![image](https://github.com/Lior-Baruch/AI_For_HealthCare/assets/80769137/17470634-809b-4107-8e27-09642e542e9f)
- ![image](https://github.com/Lior-Baruch/AI_For_HealthCare/assets/80769137/ef3b5fe1-ee40-47e6-bb7f-86ed14bf1a56)
- ![image](https://github.com/Lior-Baruch/AI_For_HealthCare/assets/80769137/63379ed0-96bf-4540-9e98-b52997a430b5)




Sample images were also visualized to understand the quality and variations present in the dataset.

- (example images from the dataset)
- ![image](https://github.com/Lior-Baruch/AI_For_HealthCare/assets/80769137/2f1b459c-3872-4b4d-b31c-be33ef55c5e3)


## Modeling Approach and Results

### Baseline Model

A Convolutional Neural Network (CNN) served as the baseline model. This model, consisting of three convolutional layers with ReLU activations and max pooling, followed by two fully connected layers, was trained using binary cross-entropy loss as the optimization criterion. After training for ten epochs, the baseline model achieved a final training accuracy of 91.43% and a validation accuracy of 64.68%.

- (plot showing the training and validation loss and accuracy over epochs for the baseline model)
- ![image](https://github.com/Lior-Baruch/AI_For_HealthCare/assets/80769137/7b8924ba-5d7c-4d53-9abe-87fe4cd3f157)


### Advanced Approach - BYOL

The advanced approach involved a Self-Supervised Learning (SSL) method known as Bootstrap Your Own Latent (BYOL). This model uses two neural networks, an online network and a target network, which aim to be as similar as possible for the same image under different augmentations, trained for 10 epochs.


### BYOL Classifier Model

Following the BYOL model training, a classifier model was trained on the learned representations. This model, a Multi-Layer Perceptron (MLP), takes the BYOL representations as input and outputs a probability distribution over the two classes. This classifier model was trained for ten epochs and achieved a final training accuracy of 74.69% and a validation accuracy of 74.36%.

- (plot showing the training and validation loss and accuracy over epochs for the classifier model)
- ![image](https://github.com/Lior-Baruch/AI_For_HealthCare/assets/80769137/10262947-4c0f-41c3-8c3b-630d0d324171)


### Comparison of Models

When comparing the three models, it is evident that the classifier model trained on BYOL representations outperforms the baseline model. The baseline model achieved a test accuracy of 53.16%, while the classifier model reached a test accuracy of 69.02%. This represents an improvement of approximately 14.65 percentage points.

- (Insert a table or plot comparing the test accuracies of the models)
- ![image](https://github.com/Lior-Baruch/AI_For_HealthCare/assets/80769137/9c9b9d3e-8c54-4cfd-b239-74e0243bbac6)
- ![image](https://github.com/Lior-Baruch/AI_For_HealthCare/assets/80769137/79dff9ae-c90c-4fb8-a0eb-858b911a11dd)



## Conclusion and Future Work

The advanced approach involving BYOL model training was computationally more expensive due to its complex architecture. However, the improvement in accuracy indicates that the additional computational effort can be justified.

Future work could involve more sophisticated data augmentation techniques, tuning of hyperparameters, and application of ensemble methods. Additionally, fine-tuning the BYOL model on a downstream task using labeled data could potentially lead to further improvements. This study demonstrates the effectiveness of Self-Supervised Learning (SSL) methods like BYOL for representation learning, particularly in scenarios where labeled data may be scarce or expensive to obtain.
