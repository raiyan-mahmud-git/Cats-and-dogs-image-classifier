# Cat vs Dog Image Classifier

## Requirements 
tensorflow>=2.10
numpy
matplotlib

This project is a Convolutional Neural Network (CNN) built using TensorFlow and Keras to classify images of cats and dogs.

It was developed as part of the freeCodeCamp Machine Learning Certification.


## Dataset

The dataset structure:

cats_and_dogs/
├── train/
│   ├── cats/
│   └── dogs/
├── validation/
│   ├── cats/
│   └── dogs/
└── test/
    └── images (unlabeled)

- Train: used for learning  
- Validation: used to evaluate during training  
- Test: used for final predictions  

Note: Dataset is not included due to size.



## Installation

Clone the repository and install dependencies:

git clone <your-repo-link>
cd cat-dog-classifier
pip install -r requirements.txt



## Usage

1. Place dataset in the correct folder structure  
2. Open the notebook in Google Colab or locally  
3. Run all cells step by step  

Optional: Load saved model

from tensorflow.keras.models import load_model
model = load_model("cat_dog_classifier_model.h5")



## Model Architecture

- Conv2D + MaxPooling layers for feature extraction  
- Flatten layer to convert image to vector  
- Dense layers for classification  
- Dropout (0.5) to prevent overfitting  
- Output layer uses sigmoid activation  



## Training Details

- Optimizer: Adam  
- Loss: Binary Crossentropy  
- Metrics: Accuracy  
- Epochs: 20  
- Batch size: 128  

Data augmentation includes:
- Rotation  
- Shifting  
- Zoom  
- Horizontal flip  



## Results

- Achieved approximately 74% accuracy  
- Successfully passed the project requirement  



## License

This project is open-source and available for learning and personal use.
