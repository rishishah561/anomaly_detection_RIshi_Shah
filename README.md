# Anomaly Detection in Data Streams

This project implements a real-time anomaly detection system for continuous data streams using Python. It uses a sliding window approach with Z-score statistics to identify anomalies in the data.

## Features

- Data stream generation with configurable parameters
- Real-time anomaly detection using Z-score statistics
- Live visualization of the data stream and detected anomalies
- Comprehensive final analysis with multiple visualizations
- Performance metrics calculation (Precision, Recall, F1 Score)

## Requirements

- Python 3.7+
- Required packages:
  - numpy
  - matplotlib
  - scipy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/rishishah561/anomaly_detection_Rishi_Shah
   cd anomaly-detection-streams
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install numpy matplotlib scipy
   ```

## Usage

Run the main script:

```
python anomaly_detection.py
```

This will:
1. Generate a synthetic data stream
2. Perform real-time anomaly detection with live visualization
3. Display a comprehensive final analysis
4. Print performance metrics

## Customization

You can modify the following parameters in the `anomaly_detection.py` file:

- In the `if __name__ == "__main__":` block:
  - `data_length`: Length of the generated data stream
  - `anomaly_probability`: Probability of anomaly occurrence
  - `trend_factor`: Strength of the upward trend in the data

- In the `AnomalyDetector` class initialization:
  - `window_size`: Size of the sliding window for Z-score calculation
  - `z_threshold`: Z-score threshold for anomaly detection

## Output

The script produces:

1. A real-time plot showing the data stream and detected anomalies as they occur.
2. A final analysis with three subplots:
   - Data stream with detected and true anomalies
   - Anomaly detection performance (true positives, false positives, false negatives)
   - Cumulative anomaly count over time
3. Performance metrics printed in the console (Precision, Recall, F1 Score)

## Algorithm Explanation

This project uses a sliding window approach with Z-score statistics for anomaly detection:

1. Maintain a sliding window of recent data points.
2. For each new data point, calculate its Z-score based on the current window.
3. If the absolute Z-score exceeds a predefined threshold, flag the point as an anomaly.

This approach allows for adaptive anomaly detection that can handle concept drift and seasonal variations in the data stream.

## License

This project is open-source and available under the MIT License.


## Author
[rishishah561]()
