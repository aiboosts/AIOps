import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import matplotlib.pyplot as plt

# Fetch CPU usage data from Prometheus
prometheus_url = 'http://localhost:9090/api/v1/query'
query = 'container_cpu_usage_seconds_total'

response = requests.get(prometheus_url, params={'query': query})
data = response.json()

# Extract and prepare data for machine learning
cpu_usage = [float(item['value'][1]) for item in data['data']['result']]

# Clean the data: Remove outliers using z-scores
cpu_usage = np.array(cpu_usage)
cpu_usage = cpu_usage[~(np.abs(stats.zscore(cpu_usage)) > 3)]  # Remove outliers

# Moving average smoothing
window_size = 5
smoothed_data = np.convolve(cpu_usage.flatten(), np.ones(window_size)/window_size, mode='valid')

# Data preparation
cpu_usage = smoothed_data.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler()
cpu_usage_normalized = scaler.fit_transform(cpu_usage)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(cpu_usage_normalized, cpu_usage_normalized, test_size=0.2)

# Build a more complex model with additional layers and regularization
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_dim=1, activation='relu', kernel_regularizer=l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    tf.keras.layers.Dense(1)
])

# Compile and train the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=MeanSquaredError())

history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32)

# Save the trained model
model.save('cpu_usage_model.h5')

# Evaluate the model
test_loss = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)

# Plot learning curves (training and validation loss)
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# Load the model
loaded_model = tf.keras.models.load_model('cpu_usage_model.h5')

# Make predictions with the loaded model
predictions = loaded_model.predict(X_test)

# Convert predictions back to actual values using the scaler
predictions_actual = scaler.inverse_transform(predictions)

# Output the predictions
print("Predictions (normalized):", predictions)
print("Predictions (actual values):", predictions_actual)

# Optional: If you want to predict on new CPU usage data
def predict_cpu_usage(new_cpu_usage):
    # Normalize the new data
    new_cpu_usage_normalized = scaler.transform(np.array(new_cpu_usage).reshape(-1, 1))
    # Make predictions
    prediction = loaded_model.predict(new_cpu_usage_normalized)
    # Convert back to actual values
    prediction_actual = scaler.inverse_transform(prediction)
    return prediction_actual

# Example of predicting with new CPU usage data
# new_data = [0.1, 0.25, 0.35]  # Example new data points
# predictions_for_new_data = predict_cpu_usage(new_data)
# print("Predictions for new data:", predictions_for_new_data)
