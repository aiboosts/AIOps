import requests
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Prometheus API URL
prometheus_url = os.getenv('PROMETHEUS_URL', 'http://localhost:9090/api/v1/query')  # Default URL if the env variable is not set
query = 'rate(container_cpu_usage_seconds_total[5m])'  # Improved metric for CPU load

# Fetch data from Prometheus
response = requests.get(prometheus_url, params={'query': query})
data = response.json()

# Extract CPU usage and pod names (only from the "default" namespace)
cpu_usage = []
pod_names = []

for item in data.get('data', {}).get('result', []):
    namespace = item.get('metric', {}).get('namespace', '')  # Retrieve namespace
    if namespace == 'default':  # Only process pods in the "default" namespace
        value = item.get('value')
        if value and len(value) > 1:
            cpu_usage.append(float(value[1]))  # Convert string to float
            pod_names.append(item.get('metric', {}).get('pod', 'unknown'))  # Extract pod name

# Handle case when no relevant data is found
if not cpu_usage:
    print("No relevant CPU data found for namespace 'default'.")
    exit()

# Prepare data
cpu_usage = np.array(cpu_usage, dtype=np.float32).reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler()
cpu_usage_normalized = scaler.fit_transform(cpu_usage)

# Load the trained model and make predictions
loaded_model = tf.keras.models.load_model('cpu_usage_model.h5')
predictions = loaded_model.predict(cpu_usage_normalized)

print(predictions)

# Output pods with high CPU prediction (Threshold: 0.8)
high_cpu_pods = []
for pod, prediction in zip(pod_names, predictions):
    if prediction[0] > 0.3 and "cpu-usage-prediction" not in pod and "prometheus" not in pod:
        print(f"Pod with high CPU usage in namespace 'default': {pod}")
        high_cpu_pods.append(pod)

# Save the list of high CPU pods to pod_list.txt
if high_cpu_pods:
    with open("/tmp/pod_list.txt", "w") as file:
        for pod in high_cpu_pods:
            file.write(f"{pod}\n")
    print(f"List of pods to delete written to /tmp/pod_list.txt")
else:
    print("No pods with high CPU usage found.")
