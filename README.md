# Evidence Detection
This repository contains two different approaches(models) to classify whether a given piece of text (evidence) supports a claim.
 
 ### Developers
 * Kushagra Srivastava
 * Saksham Goel

## Bi-LSTM Network
This model is a Bidirectional LSTM neural network designed to classify text sequences, specifically determining whether a given piece of evidence supports a claim. It leverages bidirectional processing of text for comprehensive context capture and employs layers like Dropout and Batch Normalization to enhance performance and generalization.

### Model Architecture and Inspiration
**Referenced Papers**
* [(Liu & Guo, 2019)](https://consensus.app/papers/bidirectional-lstm-attention-mechanism-layer-text-liu/29833e4a55095bd4b3ce33cf508bf796/)
* [(Xu, Xie, & Xiao, 2018)](https://consensus.app/papers/bidirectional-lstm-approach-word-embeddings-sentence-xu/60d21e7d6d895db49f82e74214851da8/)
* [(Zou & Li, 2021)](https://consensus.app/papers/lz1904-semeval2021-task-bilstmcrf-toxic-span-detection-zou/85ccd0060d9a5a74aff05c9fb8cd434c/)

These papers were referenced to gauge the performance of Bi-LSTM models in tasks such as classification tasks,effectiveness in understanding sentence relationships,etc.

**Documentations Used**
* [Tensorflow Keras](https://www.tensorflow.org/api_docs/python/tf/keras)

### Notebooks Explanation
* [Bi_LSTM.ipynb](https://github.com/SakshamGoelUK/COMP34812-NLU/blob/main/Bi_LSTM.ipynb): This is the main notebook for the Bi-LSTM model. This includes the preprocessing, model hyperparameter selection,training , and evaluation on the validation set.
* [Bi_LSTM_Demo_Code.ipynb](https://github.com/SakshamGoelUK/COMP34812-NLU/blob/main/Bi_LSTM_Demo_Code.ipynb): This is the demo code that can be used to test the model, using csv inputs or individual sentence inputs. This was used to generate the test dataset predictions
* [**Bi_LSTM_evaluation_code.ipynb**](https://github.com/SakshamGoelUK/COMP34812-NLU/blob/main/Bi_LSTM_evaluation_code.ipynb): This is the evaluation notebook that was used to test the performance of the Bi-LSTM model. This involved calculating the accuracy,precision,recall,f1-score as well as visualising data such as by using a confusion matrix
### Running the Demo Code

The demo code already sets up the file paths to the Bi-LSTM model, the trained tokenizer and the max sequence length for a **Google Colab** environment.To use an alternate environment, utilise the links given to below to download the necessary files

  * **Location of the Bi-LSTM model:** https://drive.google.com/file/d/1XG7qxfnAtyv6UskhgIHZOZbmoE6tgsG_/view?usp=drive_link
  * **Location of the Tokenizer:** https://drive.google.com/file/d/1KC38mTWjAt3EpioHNSUny4Zi3AYRQlaf/view?usp=drive_link
  * **Location of the Max Sequence Length:** https://drive.google.com/file/d/1mc5KqIRi6vtmpzGWw1zb6YKdJh2m-8AK/view?usp=drive_link

To use an alternate environment, utilise the links given above to download the necessary files and ensure that the file paths are configured correctly.
* Set `model_path` to the Bi-LSTM model file.
* For the tokenizer and max sequence length, change the paths of `handle` and `f` respectively to the appropriate files.
  For example, if your tokenizer is saved at a relative path `/files/tokenizer.pickle`,use the following:

  ```{python}
  with open('/files/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
  ```
* Set `path` to the path to the test dataset .csv file.
* If you want to save the prediction csv file to an alternate location such as to `/files/results.csv`, then make these changes in the last line of code of the `predict_from_csv` function:
  ```{python}
  predictions_df.to_csv('/files/results.csv', index=False)
  ```
NOTE: If you are not using the google colab environment , comment the fist two lines of code in the first code box
```{python}
from google.colab import drive
drive.mount('/content/drive')
```
#### Demo Code Explanation
The Demo code provides two main functions for a smooth testing experience with the trained Bi-LSTM model:
* **Function 1: predict_from_csv**

  This function can be used to get predictions for a bunch of claims and evidence pairs listed in a .csv format.It takes the path to the test dataset as a parameter. The function calls the `load_and_preprocess_data` function which assumes that the data is structured in the same way as the test dataset.
* **Function 2: predict_claim_evidence**
    This function takes in two parameters: claim and evidence . The parameters are treated as strings and the function outputs whether the claim is true or not using the Bi-LSTM model.
#### Software Requirements
*  Tensorflow 2.15.0
## Evidence Detection Model - RoBERTa

This repository contains a fine-tuned RoBERTa model for the task of evidence detection. The model classifies whether a given piece of text (evidence) supports a claim. It is a supervised model trained in English on a dataset of claim-evidence pairs.

### Model Architecture and Inspiration

- **Repository**: [RoBERTa Base - Hugging Face Model Hub](https://huggingface.co/FacebookAI/roberta-base)
- **Training Paper**: [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)

#### Additional Repositories Used

- **How to fine-tune RoBERTa using PyTorch**: [Fine-tuning RoBERTa for topic classification](https://medium.com/@achillesmoraites/fine-tuning-roberta-for-topic-classification-with-hugging-face-transformers-and-datasets-library-c6f8432d0820)
- **Intro to RoBERTa**: [A Gentle Introduction to RoBERTa](https://www.analyticsvidhya.com/blog/2022/10/a-gentle-introduction-to-roberta/)

Both these repositories were used to understand how to write the code to tune RoBERTa models used PyTorch,  and dealing with data.

### Notebooks explanation

- [**RoBERTa_tuning.ipynb**](https://github.com/SakshamGoelUK/COMP34812-NLU/blob/main/RoBERTa_tuning.ipynb): This is the main development notebook for the RoBERTa model. The preprocessing, hyperparameter selection, and training of the model is carried out in this notebook.
- [**RoBERTa_DEMO.ipynb**](https://github.com/SakshamGoelUK/COMP34812-NLU/blob/main/RoBERTa_DEMO.ipynb): This is the demo code that can be used to test the model, using csv inputs or individual sentence inputs. It was used to test on the development dataset throughout the development of the model, and was also used to generate the test dataset predictions.
- [**RoBERTa_Evaluation.ipynb**](https://github.com/SakshamGoelUK/COMP34812-NLU/blob/main/RoBERTa_Evaluation.ipynb): This is the evaluation notebook used to calculate evaluation metrics such as Accuracy, Precision, Recall, F1, but I also tried some different metrics that I deemed appropriate. There was a severe class imbalance in the validation dataset so I used the PR-AUC curve, which I had read as being appropriate in such scenarios. I also plotted the confusion matrix and calculated the Matthews Correlation Coefficient, to get an overall idea of how well the model was doing. The results were good and the model performed well across all metrics.

### Running the Demo Code

##### Location of Fine-Tuned RoBERTa model: https://drive.google.com/drive/folders/17FtIWoRF8fjc0v9xHZGpzN2T1Fg4_UVJ?usp=sharing

Download the folder at this link, and follow the steps below.
NOTE: The `model_path` must point to the folder, rather than the safetensors file, for it to be imported correctly.
To run the model and make predictions, ensure that you have the necessary paths configured for your environment:

1. Set `model_path` to the directory where the model and tokenizer are saved. (The model folder downloaded from the link above)
2. The `predict` function takes two strings - a claim and evidence - and outputs a binary prediction. You can use this function if you want to test a specific example and see the output.
3. To predict on a new CSV input file, set `link` to the path of your CSV file. The CSV should contain two columns: `Claim` and `Evidence`. The output will be a new CSV file with the predictions, with one column `prediction`.
4. Set `out` to the desired output location.

Make sure to update the path placeholders with the appropriate file paths for your specific setup.

### Software Requirements

- **Transformers**: 4.18.0
- **PyTorch**: 1.11.0+cu113



