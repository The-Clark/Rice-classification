#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Loading dataset into dataframe
df = pd.read_csv('/Users/TAM/Desktop/Arogundade/Submission/filtered_rice.csv')
df.head()


# In[3]:


df.columns = df.columns.str.lower()


# In[4]:


df.head()


# In[5]:


print(df.isnull().sum())


# In[6]:


# Explore the basic statistics of the features
df.describe()


# In[7]:


# Visualize the distribution of each feature
plt.figure(figsize=(12, 8))
sns.histplot(data=df, bins=30)
plt.title("Distribution of Features")
plt.savefig('dist.jpg', format='jpg')
plt.show()


# In[8]:


# Box plot to identify outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.title("Box Plot showing Outliers")
plt.savefig('box.jpg', format='jpg')
plt.show()


# In[9]:


# Bar chart to visualize the distribution of classes
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='class')
plt.title("Distribution of Rice Varieties")
plt.savefig('bar_plot.jpg', format='jpg')
plt.show()


# In[10]:


# Function to apply dynamic trimming based on z-scores
def trim_outliers(series, threshold):
    z_scores = np.abs((series - series.mean()) / series.std())
    return series[z_scores <= threshold]

# Set the threshold for trimming (e.g., 3, 4, or 5, depending on the desired level of trimming)
threshold = 3

# Apply dynamic trimming to 'area' and 'convex_area' columns
df['area'] = trim_outliers(df['area'], threshold)
df['convex_area'] = trim_outliers(df['convex_area'], threshold)


# In[11]:


df.head()


# In[12]:


# Let's separate the features from the target variable
X = df.drop('class', axis=1)
y = df['class']

# Filling NaNs with the mean value of the column
X = X.fillna(X.mean())

# And for the target variable
y = y.fillna(y.mode()[0])  # filling with the most frequent value as it's probably categorical


forest = RandomForestClassifier(random_state=42)
forest.fit(X, y)

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print(f"{f + 1}. feature {X.columns[indices[f]]} ({importances[indices[f]]})")
    
# Create a horizontal bar plot for feature importances
plt.figure(figsize=(10, 6))
plt.barh(range(X.shape[1]), importances[indices], align='center')
plt.yticks(range(X.shape[1]), X.columns[indices])
plt.xlabel('Feature Importance')
plt.title('Feature Importance Ranking')
plt.tight_layout()
plt.savefig('feature_rank.jpg', format='jpg')
plt.show()


# In[13]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[14]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)

y_pred_rf = clf_rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))


# In[15]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Your model
clf_rf_val = RandomForestClassifier(n_estimators=100, random_state=42)

# 10-fold cross-validation
scores = cross_val_score(clf_rf, X, y, cv=10)
print("Average Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[16]:


from sklearn.svm import SVC

clf_svm = SVC(kernel='linear', random_state=42)
clf_svm.fit(X_train, y_train)

y_pred_svm = clf_svm.predict(X_test)
print(classification_report(y_test, y_pred_svm))


# In[17]:


from xgboost import XGBClassifier

clf_xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
clf_xgb.fit(X_train, y_train)

y_pred_xgb = clf_xgb.predict(X_test)
print(classification_report(y_test, y_pred_xgb))


# In[18]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Train XGBoost on the training data
clf_xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
clf_xgb.fit(X_train, y_train)

# Generate new features from the XGBoost model
X_train_transformed = clf_xgb.apply(X_train)
X_test_transformed = clf_xgb.apply(X_test)

# Concatenate the new features to the original ones
X_train_hybrid = np.hstack((X_train, X_train_transformed))
X_test_hybrid = np.hstack((X_test, X_test_transformed))

# Convert targets to one-hot encoding for neural network
y_train_nn = tf.keras.utils.to_categorical(y_train, num_classes=5)
y_test_nn = tf.keras.utils.to_categorical(y_test, num_classes=5)

# Define and train a simple neural network
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train_hybrid.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(5, activation='softmax')) # Assuming 5 classes for rice

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train_hybrid, y_train_nn, epochs=20, batch_size=32, validation_data=(X_test_hybrid, y_test_nn))

# Generate predictions using the trained neural network model
y_pred_nn = model.predict(X_test_hybrid)
y_pred_nn_classes = np.argmax(y_pred_nn, axis=1)  # Convert probabilities to class labels

# Convert one-hot encoded labels back to original class labels
y_test_nn_classes = np.argmax(y_test_nn, axis=1)

# Calculate accuracy
accuracy_nn = accuracy_score(y_test_nn_classes, y_pred_nn_classes)
print("Accuracy (Neural Network):", accuracy_nn)

# Other evaluation metrics
print("\nClassification Report (Neural Network):\n", classification_report(y_test_nn_classes, y_pred_nn_classes))

# Visualize training error loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Error Loss')
plt.savefig('training_loss.jpg', format='jpg')
plt.show()


# In[19]:


import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.preprocessing import StandardScaler
# Step 1: Preparing the Data
df= df.dropna()
# Load your dataset
X = df.drop('class', axis=1)
y = df['class']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Step 2: Feature Extraction using CNN
# Initialize and train the CNN model
model = Sequential()
model.add(Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history= model.fit(np.expand_dims(X_train, axis=2), y_train, epochs=20)

# Extract the feature vectors
extracted_features_train = model.predict(np.expand_dims(X_train, axis=2))
extracted_features_test = model.predict(np.expand_dims(X_test, axis=2))

# Step 3: Feature Combination and Training using SVM
# Combine the extracted features with original features
X_train_combined = np.hstack((X_train, extracted_features_train))
X_test_combined = np.hstack((X_test, extracted_features_test))

# Initialize and train the SVM model
svm_model = svm.SVC()
svm_model.fit(X_train_combined, y_train)

# Evaluate the hybrid model
y_pred = svm_model.predict(X_test_combined)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


# In[22]:


import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from tabulate import tabulate

# Assuming you have already trained the models: clf_rf, clf_svm, clf_xgb
# Assuming you have made predictions for hybrid models: y_pred_hybrid_cnn_svm, y_pred_hybrid_xgb_nn

# Calculate accuracy for traditional models
y_pred_rf = clf_rf.predict(X_test)
y_pred_svm = clf_svm.predict(X_test)
y_pred_xgb = clf_xgb.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

# Calculate accuracy for hybrid models
accuracy_hybrid_cnn_svm = accuracy_score(y_test, y_pred)
accuracy_hybrid_xgb_nn = accuracy_score(y_test_nn_classes, y_pred_nn_classes)

# Create a dictionary to store the accuracy scores for all models
reports = {
    'Random Forest': accuracy_rf,
    'SVM': accuracy_svm,
    'XGBoost': accuracy_xgb,
    'Hybrid CNN-SVM': accuracy_hybrid_cnn_svm,
    'Hybrid XGBoost-NN': accuracy_hybrid_xgb_nn
}

# Convert the dictionary to a list of lists for the table
table_data = [[model, accuracy] for model, accuracy in reports.items()]

# Print the results table in a proper tabular format
headers = ['Model', 'Accuracy']
print(tabulate(table_data, headers=headers, tablefmt='grid'))

# Create a bar plot for the results
models = [model for model, _ in table_data]
accuracies = [accuracy for _, accuracy in table_data]

# Create a horizontal bar plot for feature importances
plt.figure(figsize=(12, 8))  # Increase figure size for better visibility
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)  # Adjust bottom margin
plt.bar(models, accuracies)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy for Models')
plt.xticks(rotation='vertical')  # Change rotation to vertical

# Save the bar plot as a JPG image
plt.savefig('models_accuracy.jpg')


# In[23]:


import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Setup the classifier
clf_xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# Define the hyperparameters and their possible values
param_grid = {
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.3],
    'max_depth': [3, 4, 5, 6, 7],
    'min_child_weight': [1, 2, 3, 4],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'n_estimators': [100, 200, 300, 400, 500]
}
# Initialize GridSearchCV
grid_search = GridSearchCV(clf_xgb, param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best Parameters:", grid_search.best_params_)

# Predict using the best model
y_pred_xgb = grid_search.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred_xgb))


# **IMPLEMENTATION OF AUTOML(TPOTClassifier)**

# In[26]:


from tpot import TPOTClassifier

# creating instance of TPOT
pipeline_optimizer = TPOTClassifier()


# In[27]:


pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,
                                    random_state=42, verbosity=2)

pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export('tpot_exported_pipeline.py')


# In[36]:


from tpot_exported_pipeline import exported_pipeline


# In[37]:


exported_pipeline.fit(X_train, y_train)
predictions = exported_pipeline.predict(X_test)


# In[38]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, predictions)

# Visualization of the confusion matrix using seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[39]:


print("Accuracy Score:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))


# In[ ]:




