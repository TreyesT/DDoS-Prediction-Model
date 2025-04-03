import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('../data/50k_50k.csv')
complete_df = pd.read_csv('../data/final_dataset.csv')

# , 'Fwd Seg Size Min', 'Init Bwd Win Byts'
cols_to_exclude = ['Unnamed: 0', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.difference(cols_to_exclude)
new_df = df.dropna()

for col in new_df.select_dtypes(include=[np.number]).columns:
    threshold = np.percentile(new_df[col].dropna(), 99)
    new_df.loc[new_df[col] > threshold, col] = threshold

X = new_df[numeric_columns]
y = new_df['Label']

# If necessary, encode the target
# For example, if y contains strings like 'ddos' and 'benign':
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# --- Decision Tree ---
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)
y_pred_dt = dt_clf.predict(X_test)

# --- TESTING WITH FULL CSV ---
# --------------------------
# Evaluate on Complete Dataset
# --------------------------

# Preprocess the complete dataset similar to the training data:
# Drop any rows with missing values
complete_df_clean = complete_df.dropna()

# Cap extreme values for numeric columns
for col in complete_df_clean.select_dtypes(include=[np.number]).columns:
    threshold = np.percentile(complete_df_clean[col].dropna(), 99)
    complete_df_clean.loc[complete_df_clean[col] > threshold, col] = threshold

# Use the same numeric columns that were used for training
X_complete = complete_df_clean[numeric_columns]

# Get the target variable from the complete dataset
y_complete = complete_df_clean['Label']

# Use the already-fitted LabelEncoder to transform the target variable
y_complete = le.transform(y_complete)

# Predict using the trained Decision Tree model
y_pred_complete = dt_clf.predict(X_complete)

# Evaluate the model on the complete dataset
print("Evaluation on the complete dataset:")
print("Classification Report:\n", classification_report(y_complete, y_pred_complete))
print("Accuracy:", accuracy_score(y_complete, y_pred_complete))
print("Confusion Matrix:\n", confusion_matrix(y_complete, y_pred_complete))

cm = confusion_matrix(y_complete, y_pred_complete)

custom_labels = ['Benign', 'DDoS']
# Create a heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=custom_labels, yticklabels=custom_labels)
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
plt.show()
