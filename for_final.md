Final Model Results (on held-out test set):
- Accuracy: 76.43%
- F1-Score (Macro): 76.59%
- AUC: 94.07%
- Precision: 77.02%
- Recall: 76.43%
- Loss: 1.1603
- Evaluation: The model converged well, with training loss dropping from ~2.1 (early) to ~0.637 by the end. Validation metrics improved steadily, peaking around step 11,000 (F1 ~77.6%), but showed some fluctuation in later epochs (possibly due to overfitting or early stopping not triggering).

- [x] Need to expand dataset - https://www.kaggle.com/datasets/ollypowell/new-zealand-bird-sound

- [ ] https://iclr.cc/virtual/2025/papers.html?filter=titles

Removed dropout for attention and hidden
Increased learning rate
Remove label smoothing
lora alpha not 2x
removed augmenbtation for spectrograms