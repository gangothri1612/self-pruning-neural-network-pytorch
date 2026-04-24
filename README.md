Self-Pruning Neural Network (PyTorch)
Overview

This project implements a neural network that learns to prune itself during training using learnable gate parameters. Instead of post-training pruning, the model dynamically suppresses unimportant connections during training.

Core Idea

Each weight is associated with a learnable gate:

Gate ∈ [0, 1] via sigmoid
Effective weight = weight × gate
Gate → 0 ⇒ weight is pruned
Model Architecture
Custom PrunableLinear layer
Feedforward network: 3072 → 256 → 128 → 10
Dataset: CIFAR-10
Loss Function

Total Loss = CrossEntropyLoss + λ × SparsityLoss
SparsityLoss = L1 norm of all gate values

Why L1 Encourages Sparsity

L1 regularization pushes values toward zero. Since gates are positive, minimizing their sum forces many gates toward zero, effectively pruning weights.

Experiments
Lambda	Test Accuracy	Sparsity (%)
1e-5	32.40%	0.00%
1e-4	31.45%	0.00%
1e-3	31.60%	0.00%
Observations
Accuracy slightly decreases as λ increases
Sparsity remains 0%, indicating gates did not collapse to zero
Stronger regularization or longer training may be required
Implementation is correct but pruning effect is limited in current setup
Gate Distribution

Gate values are spread between 0 and 1 with no strong spike near 0, confirming limited pruning.
