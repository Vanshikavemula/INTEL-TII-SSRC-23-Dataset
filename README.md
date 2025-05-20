# Network Traffic Classification and Analysis using TII-SSRC-23 Dataset

This repository contains a Jupyter Notebook (`ss1.ipynb`) that demonstrates a complete workflow for network traffic classification and analysis using the [TII-SSRC-23 dataset](https://www.kaggle.com/datasets/daniaherzalla/tii-ssrc-23?resource=download).

## 📘 Overview

Network traffic classification is a fundamental task in cybersecurity and networking that helps in identifying and monitoring various types of traffic for threat detection, bandwidth usage analysis, and policy enforcement. This notebook explores the TII-SSRC-23 dataset to train and evaluate a machine learning model capable of classifying different types of network traffic.

## 📁 File Structure

```
.
├── ss1.ipynb       # Jupyter Notebook for model building and analysis
└── README.md       # Project documentation
```

## 📦 Dataset

- **Name**: TII-SSRC-23: Network traffic for intrusion detection research. The TII-SSRC-23 dataset offers a comprehensive collection of network traffic patterns, meticulously compiled to support the development and research of Intrusion Detection Systems (IDS). It presents a dual structure: one part provides a tabular representation of extracted features in CSV format, while the other offers raw network traffic data for each type of traffic in PCAP files.  
- **Source**: [Kaggle Dataset Link](https://www.kaggle.com/datasets/daniaherzalla/tii-ssrc-23?resource=download)  
- **Format**: CSV

### 🔽 How to Use the Dataset

1. Visit the [dataset page on Kaggle](https://www.kaggle.com/datasets/daniaherzalla/tii-ssrc-23?resource=download).
2. Click the **Download** button to obtain the `.csv` file.
3. Place the downloaded CSV file in your local environment.
4. In the notebook (`ss1.ipynb`), locate the section where the dataset is being loaded.
5. Replace the file path in the `read_csv()` function with the path to your downloaded CSV file.
6. Run the notebook cell-by-cell for step-by-step execution.

Example:

```python
# Replace this:
df = pd.read_csv("C:\\\\Users\\\\vansh\\\\Downloads\\\\INTEL-CIC-DIS-2017-18-main\\\\data.csv\")

# With your local path:
df = pd.read_csv("your/path/here/tii-ssrc-23.csv")
```

## 🧪 Features of the Notebook

- Data loading and initial exploration  
- Preprocessing and feature engineering  
- Data visualization  
- Train-test split and normalization  
- Model training (e.g., Decision Trees, Random Forest, etc.)  
- Performance evaluation (Accuracy, Precision, Recall, F1-Score)  
- Confusion matrix and classification reports  

## 🛠️ Requirements

To run the notebook, install the following Python packages and libraries:

```bash
#install package
pip install graphviz

#install libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import pickle
```
Alternatively, you can use environments like **JupyterLab**, **Google Colab**, or **Anaconda** that come with these packages pre-installed.

## 📊 Output

Sample evaluation metrics, confusion matrix, and data visualizations are generated in the notebook to provide insights into model performance and traffic pattern recognition.

## 📌 Notes

- The dataset may be large; ensure sufficient memory before running the notebook.
- Running the entire notebook may take some time depending on the hardware and model complexity.
- This current model takes 100,000 rows as sample_input instead of whole dataset as the TII-SSRC-23 dataset is too large for the model to process 
