# import pandas as pd
# import pickle
# import matplotlib.pyplot as plt

# # The class labels
# classes = {0: 'Normal beat',
#            1: 'Supraventricular premature beat',
#            2: 'Premature ventricular contraction',
#            3: "Fusion of ventricular and normal beat",
#            4: 'Unclassifiable beat'}

# # Load the RandomForest model from the pickle file
# with open('C:/Users/Ahmad Afzal/Desktop/Medtalk Python/models/ECG.pkl', 'rb') as f:
#     model = pickle.load(f)

# # Function to load CSV, predict for a specific row, and visualize
# def predict_for_row(csv_file_path, row_number):
#     # Load the CSV file
#     data = pd.read_csv(csv_file_path, header=None)
    
#     # Separate input (features) and output (labels)
#     x_data = data.iloc[:, :-1]
#     y_data = data.iloc[:, -1]
    
#     # Get the specific row for prediction
#     single_row = x_data.iloc[row_number, :].values.reshape(1, -1)  # Reshape for prediction
#     actual_class = int(y_data.iloc[row_number])

#     # Predict the class
#     predicted_class = int(model.predict(single_row)[0])

#     # Get class labels from the dictionary
#     actual_class_label = classes.get(actual_class, "Unknown")
#     predicted_class_label = classes.get(predicted_class, "Unknown")

#     # Print the actual and predicted values
#     print(f"Prediction for row {row_number}:")
#     print(f"Predicted class: {predicted_class} ({predicted_class_label})")

#     # Visualize the row (ECG signals)
#     plt.figure(figsize=(20, 4))  # Stretch horizontally
#     plt.plot(single_row.flatten(), color='blue', label=f"ECG Signal (Row {row_number})")
#     plt.title(f"ECG Signal for Row {row_number} - Predicted Class: {predicted_class_label}")
#     plt.xlabel('Time Steps')
#     plt.ylabel('Amplitude')
#     plt.grid(True)
#     plt.legend(loc='upper right')
#     plt.show()

# # Example usage
# csv_file_path = 'mitbih_test.csv'  # Provide the path to the CSV file
# row_number = 1  # Specify the row number for prediction
# predict_for_row(csv_file_path, row_number)



from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import io
import numpy as np

app = FastAPI()

# The class labels
classes = {0: 'Normal beat',
           1: 'Supraventricular premature beat',
           2: 'Premature ventricular contraction',
           3: "Fusion of ventricular and normal beat",
           4: 'Unclassifiable beat'}

# Load the RandomForest model from the pickle file
with open('C:/Users/Ahmad Afzal/Desktop/Medtalk Python/models/ECG.pkl', 'rb') as f:
    model = pickle.load(f)

def preprocess_image_from_csv(csv_file_path, row_number):
    # Load the CSV file
    data = pd.read_csv(csv_file_path, header=None)
    
    # Separate input (features) and output (labels)
    x_data = data.iloc[:, :-1]
    y_data = data.iloc[:, -1]
    
    # Get the specific row for prediction
    single_row = x_data.iloc[row_number, :].values.reshape(1, -1)  # Reshape for prediction
    actual_class = int(y_data.iloc[row_number])

    # Predict the class
    predicted_class = int(model.predict(single_row)[0])

    # Get class labels from the dictionary
    actual_class_label = classes.get(actual_class, "Unknown")
    predicted_class_label = classes.get(predicted_class, "Unknown")

    return predicted_class_label

def create_plot(csv_file_path, row_number):
    # Load the CSV file
    data = pd.read_csv(csv_file_path, header=None)
    
    # Separate input (features) and output (labels)
    x_data = data.iloc[:, :-1]
    
    # Get the specific row for visualization
    single_row = x_data.iloc[row_number, :].values.reshape(1, -1)  # Reshape for plotting

    # Create a plot
    plt.figure(figsize=(20, 4))  # Stretch horizontally
    plt.plot(single_row.flatten(), color='blue', label=f"ECG Signal (Row {row_number})")
    plt.title(f"ECG Signal for Row {row_number}")
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend(loc='upper right')

    # Save the plot to a BytesIO object
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    plt.close()

    return img_io

@app.post("/predict")
async def predict_from_csv(file: UploadFile = File(...)):
    # Save the uploaded CSV file temporarily
    temp_csv_path = 'temp_file.csv'
    with open(temp_csv_path, 'wb') as f:
        f.write(await file.read())
    
    # Fixed row number for prediction
    row_number = 0
    
    # Get the prediction
    predicted_class_label = preprocess_image_from_csv(temp_csv_path, row_number)
    
    # Create the plot
    plot_image = create_plot(temp_csv_path, row_number)
    
    # Return the result and the plot image
    return JSONResponse(
        content={
            "predicted_class": predicted_class_label,
            "plot_url": "/plot_image"  # URL to retrieve the plot image
        }
    )

@app.get("/plot_image")
async def get_plot_image():
    temp_csv_path = 'temp_file.csv'
    row_number = 0
    
    # Create the plot image
    plot_image = create_plot(temp_csv_path, row_number)
    
    return StreamingResponse(plot_image, media_type="image/png")
