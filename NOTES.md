# Model Notes

1. **Primary signal**: Falling pitch contour and energy decay in the last 1.5s before a pause are the strongest EOT predictors; humans prosodically mark turn completion.

2. **Secondary signals**: Longer pause duration, being the final pause in a turn, and lower spectral centroid (creaky voice) all correlate with EOT.

3. **Failure modes**: The model struggles when users pause mid-sentence without prosodic completion cues, and with noisy audio where pitch tracking fails.

4. **Cross-lingual**: Hindi data was not provided for training. The model relies on universal prosodic cues, but Hindi's different intonation patterns may reduce accuracy. A Hindi-specific model would need Hindi training data.

5. **VAD limitation**: Energy-based VAD with 50dB threshold still occasionally misses or adds pauses vs. ground truth. Better VAD would improve feature-label alignment.

6. **With one more day**: Add jitter/shimmer voice quality features, speaker-normalized pitch (z-score per turn), and a validation split for threshold tuning at the 5% false-cutoff point.

7. **Causal guarantee**: Every feature uses only audio from t=0 to pause_start. The extraction window is [pause_start-1.5s, pause_start].

8. **Speed**: ~0.2s per turn on CPU; real-time capable.

9. **Calibration**: Output probabilities are uncalibrated; Platt scaling or isotonic regression on a validation set would improve threshold selection.

10. **No external data**: All features hand-engineered from raw audio. No pretrained models, no Whisper, no wav2vec, no external datasets.
