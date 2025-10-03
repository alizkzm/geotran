OverviewREST (Rank-based Estimation of Transferability) predicts how well a pre-trained model will perform on a new target task by analyzing:

Activation Shifts: Changes in feature representations between source and target domains
Weight Geometry: Intrinsic rank properties of model weights
The method combines these signals using z-score normalization to produce a unified transferability score t
