#!/usr/bin/env python3
"""
Generate EOT predictions for a data folder.
Usage: python predict.py --data_dir <folder> --out predictions.csv --model model.pkl
Output CSV columns: turn_id, pause_index, p_eot
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd

from features import load_audio, detect_pauses, extract_features


def predict(data_dir, model_path='model.pkl', out_path='predictions.csv'):
    audio_dir = os.path.join(data_dir, 'audio')
    
    if not os.path.isdir(audio_dir):
        print(f"Error: {audio_dir} not found")
        sys.exit(1)
    
    with open(model_path, 'rb') as f:
        checkpoint = pickle.load(f)
    model = checkpoint['model']
    scaler = checkpoint['scaler']
    feature_cols = checkpoint['feature_cols']
    
    predictions = []
    wav_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
    
    for wav_file in wav_files:
        turn_id = wav_file.replace('.wav', '')
        wav_path = os.path.join(audio_dir, wav_file)
        
        try:
            sr, audio = load_audio(wav_path)
        except Exception as e:
            print(f"  Skip {wav_file}: {e}")
            continue
        
        pauses = detect_pauses(audio, sr)
        duration = len(audio) / sr
        
        for i, pause in enumerate(pauses):
            feats = extract_features(audio, sr, pause['pause_start'])
            
            if feats is None:
                is_last = (i == len(pauses) - 1)
                is_long = pause['duration_ms'] > 800
                p_eot = 0.9 if (is_last and is_long) else (0.7 if is_last else 0.3)
            else:
                feats['pause_duration_ms'] = float(pause['duration_ms'])
                feats['is_last_pause'] = 1 if i == len(pauses) - 1 else 0
                feats['pause_position_ratio'] = i / max(len(pauses) - 1, 1)
                feats['turn_duration'] = float(duration)
                feats['n_pauses_in_turn'] = len(pauses)
                
                X = np.zeros((1, len(feature_cols)))
                for j, col in enumerate(feature_cols):
                    X[0, j] = feats.get(col, 0.0)
                
                X_scaled = scaler.transform(X)
                p_eot = model.predict_proba(X_scaled)[0, 1]
            
            predictions.append({
                'turn_id': turn_id,
                'pause_index': i,
                'p_eot': round(float(p_eot), 6)
            })
    
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(out_path, index=False)
    print(f"Wrote {len(predictions)} predictions to {out_path}")
    return pred_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--out', default='predictions.csv')
    parser.add_argument('--model', default='model.pkl')
    args = parser.parse_args()
    predict(args.data_dir, args.model, args.out)
