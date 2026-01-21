# Multi-task morphological tagging
We provide a pre-trained BERT model with an output layer consisting of multiple classifiers, one for each morphological feature including PoS.
Each classifier has a number of classes equal to the number of possible values for a given feature + 1 (for "None" class), except for the classifier for PoS, which does not have a "None" class.
