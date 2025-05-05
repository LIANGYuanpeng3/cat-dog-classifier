Cat vs Dog Classifier
A simple binary image classifier using PyTorch and ResNet18 to distinguish between cats and dogs.

Features
ResNet18 with transfer learning

Binary classification (cat/dog)

Trained on a subset of the Microsoft Cats vs Dogs dataset

Usage
1. Install dependencies
pip install -r requirements.txt

2. Train the model
python src/train.py

3. Run inference
Place a test image (e.g., test.jpg) in the project root, then run:
python predict.py

Dataset
Kaggle - Microsoft Cats vs Dogs
