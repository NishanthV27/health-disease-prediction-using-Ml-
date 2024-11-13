import gradio as gr
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data  # Features (30 features)
y = data.target  # Target labels (0 for malignant, 1 for benign)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a machine learning pipeline with standard scaling and SVC model
model = make_pipeline(StandardScaler(), SVC())

# Train the model
model.fit(X_train, y_train)

# Prediction function that takes user input and returns the prediction
def predict_cancer(*input_data):
    # Convert the inputs into a format the model can use (a list of values)
    prediction = model.predict([input_data])
    # Return the result: "Malignant" (0) or "Benign" (1)
    return "Malignant" if prediction == 0 else "Benign"

# Create sliders for each of the 30 features in the dataset
feature_names = data.feature_names  # Feature names from the dataset

# Create Gradio sliders for each feature
inputs = [gr.Slider(minimum=0, maximum=100, value=50, label=feature) for feature in feature_names]

# Set up the Gradio interface
iface = gr.Interface(
    fn=predict_cancer,  # The function to call for prediction
    inputs=inputs,  # The list of sliders (input features)
    outputs="text",  # The output will be text: either "Malignant" or "Benign"
    live=True  # Updates the prediction as the user adjusts the sliders
)

# Launch the Gradio interface with a shareable link
iface.launch(share=True)
