import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from collections import deque
import matplotlib.dates as mdates
from datetime import datetime, timedelta

def generate_data_stream(length=1000, seasonal_period=100, noise_factor=0.05, anomaly_probability=0.01, trend_factor=0.05):
    time = np.arange(length)
    regular_pattern = 50 + trend_factor * time
    seasonal_pattern = 10 * np.sin(2 * np.pi * time / seasonal_period)
    noise = np.random.normal(0, noise_factor * np.std(regular_pattern), size=length)
    
    data_stream = regular_pattern + seasonal_pattern + noise
    
    # Inject anomalies
    anomalies = np.random.random(length) < anomaly_probability
    data_stream[anomalies] += np.random.uniform(20, 50, size=np.sum(anomalies))
    
    return data_stream, anomalies

class AnomalyDetector:
    def __init__(self, window_size=50, z_threshold=3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.window = deque(maxlen=window_size)
    
    def detect(self, value):
        self.window.append(value)
        if len(self.window) < self.window_size:
            return False
        
        z_scores = zscore(list(self.window))
        return abs(z_scores[-1]) > self.z_threshold

def visualize_stream_with_anomalies(data_stream, detected_anomalies, true_anomalies, start_date):
    dates = [start_date + timedelta(minutes=i) for i in range(len(data_stream))]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 16), sharex=True)
    fig.suptitle("Data Stream Analysis with Anomaly Detection", fontsize=16)
    
    # Plot the data stream and anomalies
    ax1.plot(dates, data_stream, label='Data Stream', color='blue', alpha=0.7)
    detected_indices = np.where(detected_anomalies)[0]
    ax1.scatter([dates[i] for i in detected_indices], data_stream[detected_indices], color='red', label='Detected Anomalies', zorder=5)
    true_indices = np.where(true_anomalies)[0]
    ax1.scatter([dates[i] for i in true_indices], data_stream[true_indices], color='green', marker='x', label='True Anomalies', zorder=6)
    
    ax1.set_title("Data Stream with Anomalies")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot the anomaly detection performance
    true_positive = np.logical_and(detected_anomalies, true_anomalies)
    false_positive = np.logical_and(detected_anomalies, np.logical_not(true_anomalies))
    false_negative = np.logical_and(np.logical_not(detected_anomalies), true_anomalies)
    
    ax2.fill_between(dates, 0, 1, where=true_positive, color='green', alpha=0.3, label='True Positive')
    ax2.fill_between(dates, 0, 1, where=false_positive, color='red', alpha=0.3, label='False Positive')
    ax2.fill_between(dates, 0, 1, where=false_negative, color='yellow', alpha=0.3, label='False Negative')
    
    ax2.set_title("Anomaly Detection Performance")
    ax2.set_ylabel("Detection Status")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Plot the cumulative anomaly count
    cumulative_true = np.cumsum(true_anomalies)
    cumulative_detected = np.cumsum(detected_anomalies)
    ax3.plot(dates, cumulative_true, label='True Anomalies', color='green')
    ax3.plot(dates, cumulative_detected, label='Detected Anomalies', color='red')
    ax3.set_title("Cumulative Anomaly Count")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Cumulative Count")
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Format x-axis to show dates
    plt.gcf().autofmt_xdate()
    date_formatter = mdates.DateFormatter("%Y-%m-%d %H:%M")
    ax3.xaxis.set_major_formatter(date_formatter)
    
    plt.tight_layout()
    plt.show()

def stream_and_detect_anomalies(data_stream, detector, delay=0.01, start_date=None):
    detected_anomalies = []
    if start_date is None:
        start_date = datetime.now()
    dates = [start_date + timedelta(minutes=i) for i in range(len(data_stream))]
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle("Real-time Data Stream Analysis", fontsize=16)
    
    line, = ax.plot([], [], label='Data Stream', color='blue', alpha=0.7)
    scatter = ax.scatter([], [], color='red', label='Detected Anomalies')
    ax.set_title("Data Stream with Anomalies")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    date_formatter = mdates.DateFormatter("%Y-%m-%d %H:%M")
    ax.xaxis.set_major_formatter(date_formatter)
    
    plt.tight_layout()

    for i, value in enumerate(data_stream):
        is_anomaly = detector.detect(value)
        detected_anomalies.append(is_anomaly)
        
        if is_anomaly:
            print(f"Anomaly detected at {dates[i]}: value {value:.2f}")
        
        line.set_data(dates[:i+1], data_stream[:i+1])
        anomaly_indices = [j for j, a in enumerate(detected_anomalies) if a]
        scatter.set_offsets(np.column_stack(([dates[j] for j in anomaly_indices], data_stream[anomaly_indices])))
        
        ax.relim()
        ax.autoscale_view()
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(delay)
    
    plt.ioff()
    plt.close()
    return np.array(detected_anomalies)

if __name__ == "__main__":
    # Generate data stream
    data_length = 1000
    data_stream, true_anomalies = generate_data_stream(length=data_length, anomaly_probability=0.02, trend_factor=0.02)
    
    # Initialize detector
    detector = AnomalyDetector(window_size=50, z_threshold=3.0)
    
    # Set start date for the simulation
    start_date = datetime(2023, 1, 1, 0, 0)
    
    # Perform real-time anomaly detection
    detected_anomalies = stream_and_detect_anomalies(data_stream, detector, delay=0.01, start_date=start_date)
    
    # Visualize final results
    visualize_stream_with_anomalies(data_stream, detected_anomalies, true_anomalies, start_date)
    
    # Calculate and print performance metrics
    true_positive = np.sum(np.logical_and(detected_anomalies, true_anomalies))
    false_positive = np.sum(np.logical_and(detected_anomalies, np.logical_not(true_anomalies)))
    false_negative = np.sum(np.logical_and(np.logical_not(detected_anomalies), true_anomalies))
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nPerformance Metrics:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")