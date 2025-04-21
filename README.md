# pneumonia_detection_resnet18_GradCAM
This project demonstrates that deep learning models can be effectively used for pneumonia detection from chest X-rays and can be made interpretable using explainable AI techniques such as Grad-CAM.

# Dataset
The Chest X-Ray dataset from Kaggle, which contains 5,863 labeled images categorized into "Normal" and "Pneumonia," was used. These images are further split into training, validation, and test sets. The dataset is relatively balanced but contains a slight bias toward pneumonia images.
Data Preprocessing
Each image is resized to 224x224 pixels to ensure compatibility with the ResNet18 input requirements. The pixel values are normalized using the ImageNet mean and standard deviation to align with the pretrained model’s expected input. Data augmentation techniques like horizontal flipping and random rotation were applied to make the model robust against overfitting.

# Model Architecture
We utilized ResNet18, a popular convolutional neural network architecture that is known for its residual connections, which help prevent vanishing gradients in deeper networks. The final layer of the model was modified to output two classes: Normal and Pneumonia. Dropout layers were tested to evaluate their effect on generalization.
Training Procedure
•	Loss Function: CrossEntropyLoss
•	Optimizer: Adam
•	Learning Rate: 0.0001
•	Epochs: 20
•	Batch Size: 32
Training was performed on Google Colab using a T4 GPU for faster computation. Both dropout and non-dropout configurations were trained for comparison.
Explainability with Grad-CAM
To make the predictions interpretable, Grad-CAM was integrated into the trained model. Grad-CAM helps visualize the parts of the chest X-ray that were most influential in determining the model's prediction. These visualizations are vital in confirming that the model focuses on medically relevant areas such as the lung regions.
Implementation
The entire pipeline was built using PyTorch and executed in Google Colab. The pipeline includes:
•	Data loading using ImageFolder
•	Transformations and augmentation using torchvision.transforms
•	Model loading and fine-tuning from torchvision.models
•	Grad-CAM generation from the pytorch-grad-cam library
•	Upload-and-test functionality to predict on new X-ray images with accompanying visual explanations

