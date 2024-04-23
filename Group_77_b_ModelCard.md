---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/SakshamGoelUK/COMP34812-NLU.git

---

# Model Card for y42270sg-f57965ks-ED

<!-- Provide a quick summary of what the model is/does. -->

This is a classification model that was trained to
      detect whether a claim is true using the given evidence.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is a hybrid neural network tailored for text processing, integrating both convolutional and recurrent layers. It starts with an Embedding layer, which maps textual input into a higher-dimensional vector space, with the size of this space being a tunable hyperparameter varying between 100 and 200. This is followed by one to three Conv1D layers, each configurable in terms of the number of filters and kernel size, to capture spatial hierarchies in data. The convolutional layers help in extracting salient features from the embedded text, maintaining the temporal sequence which is crucial for any text-related tasks. The inclusion of a Bidirectional LSTM layer further enhances the model's ability to capture context from both past and future states, making it particularly effective for complex language understanding tasks like sentiment analysis or document classification. The model concludes with a GlobalMaxPooling1D layer to reduce dimensionality and a dense network topology that includes dropout for regularization, culminating in a final dense layer with softmax activation designed to output probabilities for two classes. This architecture is optimized using a choice of three different optimizers—Adam, RMSprop, or SGD—each with tunable learning rates, ensuring robust learning across various textual datasets.This model is trained and validated on about 23k pieces of evidence and claims

- **Developed by:** Saksham Goel Kushagra Srivastava
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Multimodal
- **Finetuned from model [optional]:** [More Information Needed]

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** [More Information Needed]
- **Paper or documentation:** [More Information Needed]

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

23,703 claim-evidence pairs as training data

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - dropout rate: 0.5
      - embedding layer dimensions: 150
      - lstm units: 128
      - num_epochs: 4

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 61 seconds
      - duration per training epoch: 12 seconds
      - model size: 12427KB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

A subset of the development set provided, amounting to 5,927 validation pairs from development dataset.

#### Metrics

<!-- These are the evaluation metrics being used. -->


    -Accuracy: 80.66 
    -Precision: 75.88
    -Recall: 72.58
    -F1 Score: 73.88
    -MCC: 48.35


### Results

The model obtained an F1-score of 73.88% and an accuracy of 80.66%.

## Technical Specifications

### Hardware


      - RAM: at least 16 GB
      - Storage: at least 2GB,
      - GPU: V100

### Software


      - Tensorflow 2.15.0

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

The integration of convolutional and recurrent layers might focus excessively on prevalent patterns seen in training, ignoring subtler, less frequent but crucial cues in the evidence.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The hyperparameters were determined by experimentation
      with different values.
