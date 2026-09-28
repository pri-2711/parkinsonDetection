# Project Report

## Problem Statement
This project aims to detect Parkinson's disease using data from different sources such as speech, handwriting, and tremor patterns. Parkinson's disease is a neurological disorder that affects movement, voice, and fine motor control. The goal is to build a system that can identify whether a person is healthy or has Parkinson's disease using machine learning.

The project also tries to estimate disease severity using a regression model based on the Parkinson's Telemonitoring dataset. In simple words, the system helps in early screening and monitoring of Parkinson's disease.

## Understanding of the Problem / Dataset
The problem is a classification and prediction task. Some data is used to classify a person as either healthy or Parkinson's, while another dataset is used to estimate the severity of the disease.

Different types of data are used because Parkinson's disease can affect:
- Speech quality and voice features
- Handwriting patterns and motor control
- Tremor behaviour
- Overall movement severity in daily recordings

The project uses multiple data sources so that the model can learn from different signs of the disease instead of relying on just one feature type.

## Dataset
The project uses several datasets:

### 1. Tremor Dataset
This dataset contains information related to tremor behaviour. It is used for classifying whether a person is healthy or has Parkinson's disease.

### 2. Speech Dataset
This dataset contains voice-based features such as jitter, shimmer, MFCC, and other acoustic measurements. These features are useful because Parkinson's disease often changes speech quality.

### 3. Handwriting Dataset
This dataset contains movement and pressure information collected while a person draws or writes. Parkinson's disease often affects hand movement and control, which can be seen in the handwriting pattern.

### 4. Telemonitoring Dataset
This dataset contains voice recording data along with labels for motor and total UPDRS scores. It is used to estimate how severe the disease is.

Each dataset has labels or target values that help the model learn the relationship between patient data and disease condition.

## Model Justification / Understanding
The project uses multiple models because the problem is not based on one single type of signal. Each data source gives different information:

- Speech model: learns voice-related signs of Parkinson's disease
- Handwriting model: learns movement and motor control differences
- Tremor model: learns abnormal motion patterns
- Fusion model: combines outputs from all individual models to make a better final decision

This is a good idea because combining different signals usually gives better results than using only one type of data. A single model may miss patterns that another data source captures clearly.

### Classification Models
For the classification task, the system can begin with a simple and strong baseline model such as Support Vector Machine (SVM). SVM is useful when the data has clear separation between healthy and Parkinson's cases, and it performs well on structured medical datasets. It is also effective when the number of features is moderate and the dataset is not extremely large.

Another strong option for classification is XGBoost. XGBoost is a tree-based ensemble method that works very well with tabular data and can capture complex relationships between features. Parkinson's disease data often contains subtle patterns across several measurements, and XGBoost can learn these nonlinear relationships effectively. It is often better than a simple linear approach when the decision boundary is not clearly separable.

In this project, the model selection can follow a practical learning path:
1. Start with a baseline classifier such as SVM for quick comparison.
2. Test XGBoost for better nonlinear pattern learning.
3. Compare the results to choose the most reliable model for each modality.

This approach helps in understanding whether the data patterns are mostly linear or whether more complex nonlinear relationships are needed.

### Regression Models
For severity estimation, a regression model is used because the goal is not just classification. Instead of saying healthy or diseased, the model predicts a numerical severity score such as Motor UPDRS.

At the beginning, a simple Linear Regression model can be used as a baseline. This is useful to understand whether there is a roughly linear relationship between voice features and disease severity. It is easy to interpret and helps to establish a starting point.

However, medical data is often not purely linear. The relationship between voice features and UPDRS score may be more complex, so a Random Forest Regressor can be used as the next step. Random Forest can capture nonlinear relationships and interactions between multiple variables without requiring a very complex setup.

Finally, XGBoost can be used for the more advanced regression model. XGBoost is powerful for tabular medical data and can capture hidden nonlinear patterns more effectively than linear models. It often performs better when there are multiple contributing features and complex dependencies between them.

So, the regression part can progress in this way:
- Linear Regression: simple baseline and easy interpretation
- Random Forest: nonlinear comparison and better feature interaction handling
- XGBoost: stronger advanced model for nonlinear and high-performing prediction

This progression makes sense because the project starts with a simple understanding of the data and then moves toward more powerful models as the complexity of the patterns becomes clearer.

## Summary
This project is about building a smart medical screening system using machine learning. It uses speech, handwriting, and tremor data to detect Parkinson's disease and predicts disease severity from voice-based clinical data. The overall idea is to use different types of patient information together to make the diagnosis more reliable and helpful for early detection.
