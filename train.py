#!/usr/bin/env python3
"""
Train End-of-Turn detection model.
Usage: python train.py --data_dir <folder> --model model.pkl
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

from features import load_audio, detect_pauses, extract_features


def train(data_dir, model_path='model.pkl'):
    audio_dir = os.path.join(data_dir, 'audio')
    labels_path = os.path.join(data_dir, 'labels.csv')
    
    if not os.path.exists(labels_path):
        print(f"Error: {labels_path} not found")
        sys.exit(1)
    
    labels_df = pd.read_csv(labels_path)
    has_labels = 'label' in labels_df.columns
    
    all_features = []
    all_labels = []
    
    wav_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
    print(f"Found {len(wav_files)} audio files")
    
    for idx, wav_file in enumerate(wav_files):
        turn_id = wav_file.replace('.wav', '')
        wav_path = os.path.join(audio_dir, wav_file)
        
        try:
            sr, audio = load_audio(wav_path)
        except Exception as e:
            print(f"  Skip {wav_file}: {e}")
            continue
        
        duration = len(audio) / sr
        pauses = detect_pauses(audio, sr)
        
        if len(pauses) == 0:
            continue
        
        turn_labels = labels_df[labels_df['turn_id'] == turn_id]
        
        for i, pause in enumerate(pauses):
            feats = extract_features(audio, sr, pause['pause_start'])
            if feats is None:
                continue
            
            feats['pause_duration_ms'] = float(pause['duration_ms'])
            feats['is_last_pause'] = 1 if i == len(pauses) - 1 else 0
            feats['pause_position_ratio'] = i / max(len(pauses) - 1, 1)
            feats['turn_duration'] = float(duration)
            feats['n_pauses_in_turn'] = len(pauses)
            
            all_features.append(feats)
            
            if has_labels:
                row = turn_labels[turn_labels['pause_index'] == i]
                if len(row) > 0:
                    lbl = 1 if str(row.iloc[0]['label']).strip().lower() == 'eot' else 0
                else:
                    lbl = 1 if (i == len(pauses) - 1) else 0
                all_labels.append(lbl)
            else:
                all_labels.append(1 if (i == len(pauses) - 1) else 0)
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx+1}/{len(wav_files)}")
    
    if len(all_features) == 0:
        print("No features extracted!")
        sys.exit(1)
    
    feature_df = pd.DataFrame(all_features)
    feature_cols = [c for c in feature_df.columns]
    
    X = feature_df[feature_cols].fillna(0).values
    y = np.array(all_labels)
    
    print(f"\nTraining samples: {len(X)}")
    print(f"EOT rate: {np.mean(y):.1%}")
    print(f"Feature count: {len(feature_cols)}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        min_samples_leaf=5,
        random_state=42
    )
    
    if len(np.unique(y)) > 1 and len(X) >= 10:
        cv = StratifiedKFold(n_splits=min(5, len(X)//2), shuffle=True, random_state=42)
        try:
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
            print(f"CV ROC-AUC: {scores.mean():.3f} (+/- {scores.std():.3f})")
        except Exception as e:
            print(f"CV skipped: {e}")
    
    model.fit(X_scaled, y)
    
    train_pred = model.predict_proba(X_scaled)[:, 1]
    train_auc = roc_auc_score(y, train_pred)
    print(f"Train ROC-AUC: {train_auc:.3f}")
    
    importances = list(zip(feature_cols, model.feature_importances_))
    importances.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 10 features:")
    for name, imp in importances[:10]:
        print(f"  {name}: {imp:.3f}")
    
    checkpoint = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
    }
    with open(model_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"\nModel saved to {model_path}")
    return model, scaler, feature_cols


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--model', default='model.pkl')
    args = parser.parse_args()
    train(args.data_dir, args.model)
