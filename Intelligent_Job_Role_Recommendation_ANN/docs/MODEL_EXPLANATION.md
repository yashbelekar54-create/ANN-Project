# Model explanation

## Why an ANN?
An artificial neural network can learn non-linear combinations of resume signals. The MLP here is a feed-forward network: features move from input to hidden layers and then to role-specific outputs.

## Why multi-label instead of multiclass?
A person can be suitable for several jobs. Multiclass classification usually chooses one class; multi-label classification predicts each role independently. We therefore use a sigmoid output neuron for every role, rather than a single softmax output.

## Feature representation
- TF-IDF captures informative words and short phrases from resume text.
- One-hot encoding represents education and domain categories.
- Standardization puts experience on a comparable numeric scale.
The fitted preprocessor must be reused exactly during inference.

## Network computation
A dense layer computes `z = W x + b`. ReLU applies `max(0, z)`. Dropout randomly disables a fraction of activations during training to reduce overfitting. Batch normalization stabilizes intermediate activations. The final sigmoid maps each output logit to a score in [0, 1].

## Training objective
Binary cross-entropy is calculated for each role label. The optimizer (Adam) adjusts weights to reduce average loss. Early stopping restores the weights from the epoch with the best validation loss.

## From scores to roles
The network returns one score per role. The program sorts these scores, selects scores above a chosen threshold, and limits the list to top-k. The scores are not calibrated probabilities and must not be interpreted as hiring chances.

## Skill-gap logic
For each recommended role, compare the candidate's declared skills with the catalog's core skills. The difference is a suggested learning list, not a formal competency assessment.

## Known limitations
The included data is synthetic and labels are generated from skill overlap. The model may mostly learn the annotation rule. Strong test scores on this dataset do not demonstrate generalization to real resumes. Human-labelled, independently sourced and representative data is necessary for meaningful external validation.
