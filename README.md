# 🌳 Tree Species Classification

This project is about building a machine learning model to classify **tree species** using dataset features.  
The model was trained and saved as a `.keras` file for later use.


## 📌 Objective

The main objective of this project is to predict the **species of trees** based on given input features using machine learning techniques.


## 📊 Dataset

- The dataset contains samples of different tree species:
  Source:[https://www.kaggle.com/datasets/viditgandhi/tree-species-identification-dataset]
- Each record has several input features and a label representing the species.
- Data is split into **training** and **testing** sets to evaluate the model.


## 🧠 Model

- Framework: **Keras / TensorFlow**
- Model type: Neural Network
- Saved model file: `tree_species_model.keras`
- Training involved multiple epochs with accuracy and loss monitoring.



## 📝 Results

- The model achieved good accuracy on the test set.
- Evaluation metrics such as **Accuracy, Precision, Recall** were used.
- Confusion matrix and plots were generated to visualize performance.

## ⚙️ Requirements

Install dependencies before running the project:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras


▶️ How to Run

1.Clone the repository:

git clone: https://github.com/your-username/tree-species-classification.git
cd tree-species-classification


2.Run Jupyter Notebook:

jupyter notebook
Open the notebook and train the model, or directly load the saved model:
from tensorflow import keras
model = keras.models.load_model("tree_species_model.keras")

📂 Project Structure
tree-species-classification/
├── data/                 # Dataset files
├── notebooks/            # Jupyter notebooks for training/testing
├── models/
│   └── tree_species_model.keras   # Saved model
├── src/                  # Source code files
│   ├── data_preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── reports/              # Graphs and results
├── requirements.txt
└── README.md


🚀 Future Scope

1.Improve accuracy with hyperparameter tuning.

2.Try CNN/RNN models for better classification.

3.Deploy the model using Flask/Streamlit.



📄 requirements.txt
1.numpy
2.pandas
3.matplotlib
4.seaborn
5.scikit-learn
6.tensorflow
7.keras
8.jupyter

✅ Explanation

1.numpy, pandas → data handling & preprocessing.

2.matplotlib, seaborn → data visualization & plots.

3.scikit-learn → metrics, preprocessing, train-test split.

4.tensorflow, keras → building and loading the .keras model.

5.jupyter → running notebooks inside VS Code.


## 📸 Screenshots:
These screenshots illustrate key parts of the project's development and results.

1.Number of Species:
This image shows the initial data exploration, confirming the number of unique tree species in the dataset.

![Number of Species](![Images/Screenshot 1.png](<Screenshot 1.png>))

2.Model Architecture:
This is a summary of the neural network model, including the layers, output shapes, and the total number of parameters.

![Model Architecture](![Images/Screenshot 2.png](<Screenshot 2.png>))

3.Training Progress:
This screenshot displays the model's training process over multiple epochs, showing the accuracy and loss values.

![Training Progress](![Images/Screenshot 3.png](<Screenshot 3.png>))

4.Accuracy and Loss Plots:
This plot visualizes the model's performance during training, comparing the accuracy and loss for both the training and validation datasets.

![Accuracy and Loss Plots](![Images/Screenshot 4.png](<Screenshot 4.png>))
