---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/SakshamGoelUK/COMP34812-NLU

---

# Model Card for f57695ks-y42270sg-ED

<!-- Provide a quick summary of what the model is/does. -->

This is a classification model that has been trained to take two inputs where one is a statement, or 'claim,' and the other is potential 'evidence.' It's designed to evaluate whether the evidence presented is relevant and supportive of the claim made. 


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is based on a RoBERTa model that was fine-tuned on 23,703 pairs of texts.

- **Developed by:** Kushagra Srivastava and Saksham Goel
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Transformers
- **Finetuned from model [optional]:** roberta-base

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/FacebookAI/roberta-base
- **Paper or documentation:** https://arxiv.org/abs/1907.11692

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

23,703 claim-evidence pairs as training data, and 5,926 pairs as validation data

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->

or
      - learning_rate: 1e-05
      - train_batch_size: 32
      - eval_batch_size: 32
      - num_epochs: 5

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 3
      - duration per training epoch: 35 minutes
      - model size: 475.5mb

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

5,927 validation pairs from development dataset.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Precision: 85.28
      - Recall: 86.88
      - F1-score: 86.02
      - Accuracy: 88.73

### Results

The model obtained the highest metrics across all baseline models, and performed better than the other Bi-LSTM that our group developed.
                 The accuracy was 88.73% and the F1-Score was 86.02% The MCC was highest at 72.15.

## Technical Specifications

### Hardware


      - RAM: at least 16 GB
      - Storage: at least 2GB,
      - GPU: V100

### Software


      - Transformers 4.18.0
      - Pytorch 1.11.0+cu113

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Any inputs (concatenation of two sequences) longer than
      512 subwords will be truncated by the model.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The learning rate was determined by experimentation
      with different values, but due to computational limitations, we could not tune additional hyperparameters.
