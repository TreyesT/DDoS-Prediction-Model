# Predicting DDoS Attacks Using Network Flow Characteristics

## Project Overview
This repository explores whether **network flow characteristics** can accurately predict whether a flow is **benign** or a **DDoS attack**. By analyzing features such as packet sizes, inter-arrival times, flag counts, and other flow-level metrics, we build and evaluate multiple machine learning models including Decision Trees, Logistic Regression with L1 regularization (LASSO), and Gradient Boosting Classifiers. The project demonstrates challenges in handling large datasets, managing extreme values, and the importance of feature engineering for cybersecurity applications.

## File/Folder Organization

- **articles/**  
  Contains write-ups and documentation summarizing the main findings, methodologies, and insights from the analysis.

- **data/**  
  Contains a subset of the dataset used for training and evaluation.  
  - **50k_50k.csv**: A smaller sample of the network flow data used to train the models.
  - **Complete Dataset**: The full dataset (approximately 13 million rows) is too large to include in the repo. You can download it from [Kaggle: DDoS Datasets](https://www.kaggle.com/datasets/devendra416/ddos-datasets).

- **notebooks/**  
  Contains Jupyter notebooks that detail the project workflow:
  - **Model_Training.ipynb** – Data cleaning, exploratory data analysis (EDA), feature engineering, and model training.
  - **Decision_Tree_Classifier_Data_Analysis.ipynb** – In-depth analysis and visualization using the Decision Tree Classifier.
  - **Full_Data_Testing.ipynb** – Testing the trained model on the complete dataset to assess its performance on a large scale.

- **presentations/**  
  Contains presentation materials (e.g., slides, PDFs) that provide a high-level summary of the project, including key metrics, methodologies, and future directions.

## Key Findings
- **Feature Importance:**  
  Models such as Gradient Boosting and LASSO reveal that certain network flow features (e.g., flow duration, packet sizes, and flag counts) are critical in predicting DDoS attacks.
- **Model Performance:**  
  While initial evaluations on a smaller subset show very high accuracy, testing on the complete dataset indicates potential performance drops, highlighting issues like overfitting or the need for additional tuning.
- **Scalability Challenges:**  
  Processing and analyzing large-scale network data requires robust preprocessing techniques (e.g., capping extreme values) to ensure model stability and reliable performance.

## Data Sources
1. **Training Subset:**  
   - **50k_50k.csv**: A preprocessed subset of network flow data included in this repository.
2. **Complete Dataset:**  
   - Available on Kaggle: [DDoS Datasets](https://www.kaggle.com/datasets/devendra416/ddos-datasets).

## How to Use
1. **Clone or Download** this repository.
2. Explore the **notebooks/** folder in the following order:
   - **Model_Training.ipynb** – Start with data preprocessing, EDA, and initial model training.
   - **Decision_Tree_Classifier_Data_Analysis.ipynb** – Dive into detailed analysis using the Decision Tree Classifier.
   - **Full_Data_Testing.ipynb** – Evaluate the trained model on the complete dataset (13 million rows) to understand real-world performance.
3. Refer to the **articles/** and **presentations/** folders for concise summaries and visual highlights of the project.

## Future Work
- **Hyperparameter Tuning:**  
  Utilize techniques like GridSearchCV or RandomizedSearchCV to optimize model parameters for improved performance.
- **Advanced Ensemble Methods:**  
  Experiment with ensemble strategies (stacking, blending, etc.) to enhance predictive accuracy.
- **Real-time Implementation:**  
  Adapt the processing and modeling pipeline for real-time DDoS detection applications.
- **Enhanced Feature Engineering:**  
  Further explore and engineer features to capture subtle aspects of network behavior.
- **Scalability Improvements:**  
  Develop more efficient data processing pipelines to handle and analyze large-scale network traffic in real time.
