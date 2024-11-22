# %% [markdown]
# # Mritunjoy Paul  
# **NJIT UCID:** 31690401  
# **Email Address:** mp2362@njit.edu  
# **Date:** 22 Novembor 2024  
# **Professor:** Yasser Abdullah  
# **CS 634-101:** Data Mining

# %% [markdown]
# # Introduction
# This code represents my final project. The goal of this project is to use three algorithms, including Random forest, on the same data set and determine which algorithm provides better accuracy. I have taken my data set from Kaggle, and I have included a description of the data set in the folder. It's called the Census Income dataset from the UCI Machine Learning Repository. This contains information from the 1994 U.S. Census, commonly used for classification tasks like predicting income level. I have used 1000 data points for my calculation.

# %% [markdown]
# # Step 1: Installation 
# The user can install the below packages, which are required to run this code, by removing the "#" below. If you are using the Jupyter Notebook for the first time, it is recommended to install the packages.

# %%

#!pip install pandas numpy scikit-learn seaborn matplotlib
#!pip install tensorflow

# %% [markdown]
# 
# I am using an import package called warnings and os to hide the warnings from my Python code.

# %%
import warnings

warnings.filterwarnings("ignore")

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import os
print(os.getcwd())


# %% [markdown]
# # Step 2: Import Packages and Data set in the program
# 
# First, we will Import the necessary libraries. Then the data set from the environment. To use the data set, you need to keep the data set and the program in the same folder. You can get the full data set from the following link: https://archive.ics.uci.edu/dataset/2/adult

# %%

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset (assuming the file is uploaded to Colab as 'adult_data_with_headers.csv')
#data = pd.read_csv('adult_data_with_headers_Small.csv')
#data = pd.read_csv('/Users/mritunjoypaul/Desktop/Lectures and Papers/Data Mining/Final_Project/Final/Paul_Mritunjoy_finaltermproj/adult_data_with_headers_Small.csv')
import os
import pandas as pd

# Dynamically set the working directory to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Load the file from the same directory as the script
data = pd.read_csv('adult_data_with_headers_Small.csv')

# %% [markdown]
# # Step 3: Use Categorical Features
# 
# The code below prepares the dataset as categorical variables for machine learning. It encodes them into numeric representations and ensures that both the features and target variables are ready for modeling.

# %%
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    if column != 'income': 
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))
        label_encoders[column] = le

X = data.drop('income', axis=1)
y = data['income']

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# %% [markdown]
# # Step 4: Initialize K-Fold Cross-Validation
# 
# My below code will initialize 10 fold cross validation

# %%

kf = KFold(n_splits=10, shuffle=True, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
fold_accuracies = [] 

# %% [markdown]
# # Step 5: Random Forest 
# 
# Random Forest is an algorithm that builds multiple decision trees (CART) and combines their results using bagging. It is used for both classification and regression tasks. This improves accuracy and robustness compared to a single decision tree. This code performs 10-fold Cross-Validation on a dataset using a Random Forest model. It computes various performance metrics for each fold. 
# 

# %%
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_squared_error

metrics_per_fold = []

for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1] 

    BS = round(mean_squared_error(y_test, y_prob), 4)

    y_mean = y_test.mean() 
    BSS_baseline = mean_squared_error(y_test, [y_mean] * len(y_test)) 
    BSS = round(1 - BS / BSS_baseline if BSS_baseline > 0 else 0, 4)

    cm = confusion_matrix(y_test, y_pred)
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    P = TP + FN
    N = TN + FP

    Recall = round(TP / P if P > 0 else 0, 2)
    Precision = round(TP / (TP + FP) if (TP + FP) > 0 else 0, 2)
    F1 = round((2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0, 2) 
    ACC = round((TP + TN) / (P + N) if (P + N) > 0 else 0, 2)  
    TPR = round(TP / (TP + FN) if (TP + FN) > 0 else 0, 2)
    TNR = round(TN / (TN + FP) if (TN + FP) > 0 else 0, 2)
    FPR = round(FP / (FP + TN) if (FP + TN) > 0 else 0, 2)
    FNR = round(FN / (FN + TP) if (FN + TP) > 0 else 0, 2)
    Error_rate = round((FP + FN) / (TP + TN + FP + FN), 2)
    BACC = round((TPR + TNR) / 2, 2) 
    TSS = round(TPR - FPR, 2)
    HSS = round(2 * (TP * TN - FP * FN) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))
                if ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)) > 0 else 0, 2)

    metrics_per_fold.append({
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'P': P,
        'N': N,
        'Recall': Recall,
        'Precision': Precision,
        'F1_measure': F1,
        'Accuracy': ACC,
        'TPR': TPR,
        'TNR': TNR,
        'FPR': FPR,
        'FNR': FNR,
        'Error_rate': Error_rate,
        'BACC': BACC,
        'TSS': TSS,
        'HSS': HSS,
        'Brier_Score': BS,
        'Brier_Skill_Score': BSS
    })


metrics_random_forest = pd.DataFrame(metrics_per_fold).T 
metrics_random_forest.columns = [f'iter{fold + 1}' for fold in range(len(metrics_per_fold))]

metrics_random_forest = metrics_random_forest.round(2)

metrics_random_forest

# %% [markdown]
# # Step 6: Decision Tree
# 
# A Decision Tree is a type of machine-learning model used for classification and regression tasks. It splits data into branches based on decision rules derived from the input features. This will form a tree-like structure. This code performs 10-fold Cross-Validation on a dataset using an LSTM model. It computes various performance metrics for each fold.

# %%
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, mean_squared_error

dt_model = DecisionTreeClassifier(random_state=42)

metrics_per_fold = []

for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    dt_model.fit(X_train, y_train)

    y_pred = dt_model.predict(X_test)
    y_prob = dt_model.predict_proba(X_test)[:, 1]  

    BS = round(mean_squared_error(y_test, y_prob), 2)


    y_mean = y_test.mean() 
    BSS_baseline = mean_squared_error(y_test, [y_mean] * len(y_test)) 
    BSS = round(1 - BS / BSS_baseline if BSS_baseline > 0 else 0, 2)

    cm = confusion_matrix(y_test, y_pred)
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    P = TP + FN
    N = TN + FP

    Recall = round(TP / P if P > 0 else 0, 2)
    Precision = round(TP / (TP + FP) if (TP + FP) > 0 else 0, 2)
    F1 = round((2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0, 2) 
    ACC = round((TP + TN) / (P + N) if (P + N) > 0 else 0, 2) 
    TPR = round(TP / (TP + FN) if (TP + FN) > 0 else 0, 2)
    TNR = round(TN / (TN + FP) if (TN + FP) > 0 else 0, 2)
    FPR = round(FP / (FP + TN) if (FP + TN) > 0 else 0, 2)
    FNR = round(FN / (FN + TP) if (FN + TP) > 0 else 0, 2)
    Error_rate = round((FP + FN) / (TP + TN + FP + FN), 2)
    BACC = round((TPR + TNR) / 2, 2) 
    TSS = round(TPR - FPR, 2)
    HSS = round(2 * (TP * TN - FP * FN) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))
                if ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)) > 0 else 0, 2)

    metrics_per_fold.append({
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'P': P,
        'N': N,
        'Recall': Recall,
        'Precision': Precision,
        'F1_measure': F1,
        'Accuracy': ACC,
        'TPR': TPR,
        'TNR': TNR,
        'FPR': FPR,
        'FNR': FNR,
        'Error_rate': Error_rate,
        'BACC': BACC,
        'TSS': TSS,
        'HSS': HSS,
        'Brier_Score': BS,
        'Brier_Skill_Score': BSS
    })

metrics_decision_tree = pd.DataFrame(metrics_per_fold).T  
metrics_decision_tree.columns = [f'iter{fold + 1}' for fold in range(len(metrics_per_fold))]

metrics_decision_tree

# %% [markdown]
# # Step 7: Long Short-Term Memory Network (LSTM)
# 
# It's a Recurrent Neural Network (RNN) designed with special gates to store and manage both long-term and short-term memory. This will help it to avoid the vanishing gradient problem commonly found in standard RNNs. This code performs 10-fold Cross-Validation on a dataset using an LSTM model. It computes various performance metrics for each fold.

# %%

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, mean_squared_error

X = data.drop('income', axis=1)
y = data['income']

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)
y = to_categorical(y) 

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = X.reshape((X.shape[0], 1, X.shape[1]))

kf = KFold(n_splits=10, shuffle=True, random_state=42)

metrics_per_fold = []

for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
    # Split data for current fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))  # Output layer

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)  # Reduced epochs for faster cross-validation

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    BS = round(mean_squared_error(y_test[:, 1], y_pred[:, 1]), 2)

    y_mean = y_test[:, 1].mean() 
    BSS_baseline = mean_squared_error(y_test[:, 1], [y_mean] * len(y_test))
    BSS = round(1 - BS / BSS_baseline if BSS_baseline > 0 else 0, 2)

    cm = confusion_matrix(y_test_classes, y_pred_classes)
    TP = cm[1, 1] if cm.shape == (2, 2) else 0
    TN = cm[0, 0] if cm.shape == (2, 2) else 0
    FP = cm[0, 1] if cm.shape == (2, 2) else 0
    FN = cm[1, 0] if cm.shape == (2, 2) else 0

    P = TP + FN
    N = TN + FP

    Recall = round(TP / P if P > 0 else 0, 2)
    Precision = round(TP / (TP + FP) if (TP + FP) > 0 else 0, 2)
    F1 = round((2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0, 2)
    ACC = round((TP + TN) / (P + N) if (P + N) > 0 else 0, 2)
    TPR = round(TP / (TP + FN) if (TP + FN) > 0 else 0, 2)
    TNR = round(TN / (TN + FP) if (TN + FP) > 0 else 0, 2)
    FPR = round(FP / (FP + TN) if (FP + TN) > 0 else 0, 2)
    FNR = round(FN / (FN + TP) if (FN + TP) > 0 else 0, 2)
    Error_rate = round((FP + FN) / (TP + TN + FP + FN), 2)
    BACC = round((TPR + TNR) / 2, 2)
    TSS = round(TPR - FPR, 2)
    HSS = round(2 * (TP * TN - FP * FN) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))
                if ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)) > 0 else 0, 2)

    metrics_per_fold.append({
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'P': P,
        'N': N,
        'Recall': Recall,
        'Precision': Precision,
        'F1_measure': F1,
        'Accuracy': ACC,
        'TPR': TPR,
        'TNR': TNR,
        'FPR': FPR,
        'FNR': FNR,
        'Error_rate': Error_rate,
        'BACC': BACC,
        'TSS': TSS,
        'HSS': HSS,
        'Brier_Score': BS,
        'Brier_Skill_Score': BSS
    })

metrics_lstm = pd.DataFrame(metrics_per_fold).T
metrics_lstm.columns = [f'Fold_{fold + 1}' for fold in range(len(metrics_per_fold))]

metrics_lstm

# %% [markdown]
# # Step 8: Calculate and Compare Average Accuracy Across All Folds
# 
# The below table code will compare the ouputs from three algorithms 

# %%

average_rf_metrics = metrics_random_forest.mean(axis=1).round(2) 


average_dt_metrics = metrics_decision_tree.mean(axis=1).round(2)  

average_lstm_metrics = metrics_lstm.mean(axis=1).round(2)

comparison_df = pd.DataFrame({
    'Random Forest': average_rf_metrics,
    'Decision Tree': average_dt_metrics,
    'LSTM': average_lstm_metrics
})

print("Comparison of Average Metrics Across 10 Folds:")
print(comparison_df)


# %% [markdown]
# # Discussion of Results
# 
# <h1 style="color:green;">Random Forest</h1>
# 
# Random Forest show a good performance. It has an accuracy of 83% and a precision of 0.68. It slightly outperforms Decision Tree and bit behind LSTM in precision. It exhibited a TNR = 0.92 and TPR = 0.53, which shows reliable true negative and positive detection. The error rate of 0.17 and Brier score of 0.11 were among the lowest. which shows its robustness. The Brier Skill Score of 0.36 is predictive of reliability across imbalanced datasets.
# 
# 
# <h1 style="color:g;">Decision Tree</h1>
# 
# The decision tree showed the best TPR = 0.63; it detected true positives more effectively than the other models.
# It has the lowest TNR = 0.85, a higher false positive rate. Its accuracy of 80% is lower than both Random Forest and LSTM. The precision of 0.57 is a moderate performance in correctly identifying positives. The Brier score of 0.20 and a negative Brier Skill Score (-0.13) show that the Decision Tree struggled with reliability and overfitted to the training data.
# 
# 
# <h1 style="color:green;">Long Short-Term Memory Network (LSTM)</h1>
# 
# 
# LSTM achieved the highest TNR = 0.94 and the lowest FPR = 0.06, makes it the most reliable for true negative detection.
# It also showed the best precision (0.71). However, its TPR = 0.50 is the lowest, which means poorer detection of true positives. The accuracy of 84% is the highest among all models. A low Brier score (0.12) and strong Brier Skill Score (0.35), highlight its overall reliability.
# 
# <h1 style="color:blue;">Which Performed Better and Why</h1>
# 
# LSTM performed the best overall due to its highest accuracy (84%), TNR = 0.94, and precision (0.71). Its Brier Skill Score (0.35) underscores its robust predictive capabilities, even in scenarios with class imbalance. However, LSTM takes the most time to run because it is a deep learning model.  Random Forest ranked second with TNR = 0.92, precision (0.68), and TPR = 0.53. It offers a good balance between sensitivity and specificity, making it a reliable choice for various scenarios. Decision Tree ranked third, with a high FPR = 0.15 and a negative Brier Skill Score (-0.13). Its lower accuracy (80%) highlights its limitations when dealing with complex datasets and class imbalance. Both LSTM and Random Forest are effective models for prediction. With a larger dataset, Random Forest might outperform LSTM due to its ability to generalize better and its faster runtime.

# %% [markdown]
# # Github Link for the code
# https://github.com/Mritunjoy-NJIT/CS_634_Data_Mining_Final_Project

# %%



