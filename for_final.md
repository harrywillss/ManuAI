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

Considerations:
 - LoRA for smaller dataset but bigger effect and less compute
 - Lots of adaptations for better memory efficiency (to finetune model local)
   - Clears hugging face dataset cache first
 - End-to-end pipeline (download -> preprocess -> segment -> spectrogram convert -> LoRA)
 - seed for reproducibility 
 - training size choosable by user 

Class: bellbird, Segment Count: 61489
Class: morepork, Segment Count: 60698
Class: tomtit, Segment Count: 58462
Class: silvereye, Segment Count: 50446
Class: tui, Segment Count: 26785
Class: greywarbler, Segment Count: 21735
Class: kiwi, Segment Count: 11006
Class: robin, Segment Count: 9215
Class: kea, Segment Count: 4412
Class: whitehead, Segment Count: 2798
Class: pukeko, Segment Count: 1813
Class: kokako, Segment Count: 1519
Class: fantail, Segment Count: 1515
Class: kaka, Segment Count: 1486
Class: saddleback, Segment Count: 1238
Class: yellowhead, Segment Count: 773
Class: stitchbird, Segment Count: 723
Class: kingfisher, Segment Count: 313
Class: kereru, Segment Count: 291
Class: kakapo, Segment Count: 13
Overall Segment Count: 316730