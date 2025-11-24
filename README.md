<img width="3097" height="1225" alt="Picture1" src="https://github.com/user-attachments/assets/c88b3760-d088-4a9d-8601-650a92512467" />


# Brain Tumor Classification - Model Training Pipeline

A modular and reproducible deep learning pipeline for training and evaluating multiple convolutional neural networks (CNNs) on brain MRI data for tumor classification. This repository demonstrates a professional approach to medical image analysis, featuring custom-built modules for data handling, training, and visualization.

## 🚀 Key Features
### Modular Code Architecture:
Implements custom Python modules (dataloader, train_function, helper_functions) for maintainable and reusable code

### Multi-Architecture Support:
Framework designed to train and compare multiple CNN architectures (AlexNet, VGG, ResNet, GoogleNet)

### Transfer Learning Implementation:
Utilizes pre-trained weights with fine-tuning strategies for medical imaging tasks

### Comprehensive Evaluation:
Includes training metrics, visualization, and inference testing on sample images

### Google Colab Integration: 
Optimized for GPU-accelerated training in the Colab environment

## 🔧 Custom Module Specifications
### dataloader.py
Creates PyTorch DataLoaders with appropriate transformations

Handles train/test split and data augmentation

Manages class balancing and dataset statistics

### train_function.py
Implements training and validation loops with metrics tracking

Includes early stopping and model checkpointing

Manages GPU/CPU device allocation automatically

### helper_functions.py
plot_loss_curves(): Visualizes training and validation metrics

pred_and_plot_image(): Displays model predictions with ground truth comparison

Additional utilities for model evaluation and result analysis

### The dataset included glioma, meningioma, pituitary, and no-tumor MRI images

# 🚀 Installation & Setup
Clone the repository

bash
git clone https://github.com/yourusername/brain-tumor-models-training.git
cd brain-tumor-models-training
Install dependencies

bash
pip install -r requirements.txt
Mount Google Drive (for Colab)

python
from google.colab import drive
drive.mount('/content/drive')
💻 Usage Example
python
### Initialize model with transfer learning
weights = torchvision.models.AlexNet_Weights.DEFAULT
model = torchvision.models.alexnet(weights=weights)

### Freeze feature extraction layers
for param in model.features.parameters():
    param.requires_grad = False

### Custom classifier for 4-class brain tumor classification
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=4, bias=True)
)

### Train model using custom training function
results = train_function.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=10,
    device=device
)
# 
# 🔬 Methodology
Transfer Learning: Utilized pre-trained ImageNet weights with fine-tuning

Data Augmentation: Applied random transformations to improve generalization(the codes are not provided)

Feature Extraction: Frozen convolutional layers, trained only classifier heads

Optimization: Adam optimizer with cross-entropy loss function

Regularization: Dropout (p=0.2) to prevent overfitting

# 📁 Outputs
Trained model weights (.pth files)

Training/validation loss and accuracy curves

Sample predictions with visualizations

Performance metrics and model comparisons

# 🔮 Future Enhancements
Implementation of 3D CNN architectures for volumetric data

Integration of attention mechanisms for improved interpretability

Cross-validation and hyperparameter tuning automation

Model ensemble methods for improved performance

DICOM data preprocessing pipeline

# 👨‍💻 Developer
Mohammad Farhadi Rad

MSc Radiobiology | BSc Radiology Technology | AI Enthusiast
visit: https://scholar.google.com/citations?view_op=view_citation&hl=en&user=owitg_8AAAAJ&citation_for_view=owitg_8AAAAJ:u-x6o8ySG0sC

Research Interests: Quantitative Neuroimaging, AI in Medical Diagnostics, Multimodal Data Fusion
