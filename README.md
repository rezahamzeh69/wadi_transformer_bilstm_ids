# WADI Transformer-BiLSTM IDS

End-to-end pipeline for anomaly and attack detection on the WADI industrial control system dataset. The script downloads the data via KaggleHub, performs robust preprocessing and sliding-window aggregation to create labeled segments, then trains a sensor-token Transformer + BiLSTM model with attention. Outputs include train/val/test CSVs, a tuned decision threshold, a confusion matrix, and a classification report.

## Highlights
• One-file training script ready for Colab or local GPU
• Automatic dataset fetch with KaggleHub and deterministic preprocessing
• Balanced windowed re-segmentation with positive and hard-negative examples
• Sensor-wise attention Transformer+BiLSTM classifier
• Threshold tuning with a minimum-precision constraint
