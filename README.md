# WADI Transformer-BiLSTM IDS

End-to-end pipeline for anomaly and attack detection on the WADI industrial control system dataset. The script downloads the data via KaggleHub, performs robust preprocessing and UNSW-style window aggregation, then trains a sensor-token Transformer + BiLSTM model with attention. Outputs include train-val-test CSVs, tuned decision threshold, confusion matrix and classification report.

## Highlights
- One-file training script ready for Colab or local GPU
- Automatic dataset fetch with KaggleHub and safe preprocessing
- UNSW-style recordization with positive and hard-negative windows
- Sensor-wise attention Transformer+BiLSTM classifier
- Threshold tuning with a minimum precision constraint

## Quick start
```bash
# clone and enter
git clone https://github.com/<your-user>/wadi_transformer_bilstm_ids.git
cd wadi_transformer_bilstm_ids

# install
python -m pip install -r requirements.txt

# run training
python scripts/train_unswized.py
