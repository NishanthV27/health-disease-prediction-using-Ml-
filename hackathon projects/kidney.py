import pandas as pd
import numpy as np
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
file_path = 'kidney_disease.csv'
data = pd.read_csv(file_path)

# Preprocessing
data = data.drop(columns=['id'])  # Drop unnecessary columns like 'id'

# Fill missing values
for col in data.select_dtypes(include=['float64', 'int64']).columns:
    data[col].fillna(data[col].median(), inplace=True)
for col in data.select_dtypes(include=['object']).columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col])

# Separate features and target
X = data.drop(columns=['classification'])
y = data['classification']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Model evaluation (optional)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Define Gradio prediction function
def predict_kidney_disease(*inputs):
    input_data = np.array(inputs).reshape(1, -1)  # Reshape input for single prediction
    input_data = scaler.transform(input_data)  # Scale input data
    prediction = model.predict(input_data)
    return "Kidney Disease Detected" if prediction[0] == 1 else "No Kidney Disease Detected"

# Create Gradio interface
input_labels = list(X.columns)  # List of input feature names for Gradio
input_components = [gr.Number(label=label) for label in input_labels]

output_component = gr.Textbox(label="Prediction")

# Launch Gradio app
app = gr.Interface(fn=predict_kidney_disease, inputs=input_components, outputs=output_component, 
                   title="Kidney Disease Prediction",
                   description="Enter patient details to predict the likelihood of kidney disease.")

app.launch()
