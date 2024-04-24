# Evidence Detection Model - RoBERTa

This repository contains a fine-tuned RoBERTa model for the task of evidence detection. The model classifies whether a given piece of text (evidence) supports a claim. It is a supervised model trained in English on a dataset of claim-evidence pairs.

- **Developer**: Kushagra Srivastava
## Model Architecture and Inspiration

- **Repository**: [RoBERTa Base - Hugging Face Model Hub](https://huggingface.co/FacebookAI/roberta-base)
- **Training Paper**: [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)

### Additional Repositories Used

- **How to fine-tune RoBERTa using PyTorch**: [Fine-tuning RoBERTa for topic classification](https://medium.com/@achillesmoraites/fine-tuning-roberta-for-topic-classification-with-hugging-face-transformers-and-datasets-library-c6f8432d0820)
- **Intro to RoBERTa**: [A Gentle Introduction to RoBERTa](https://www.analyticsvidhya.com/blog/2022/10/a-gentle-introduction-to-roberta/)

Both these repositories were used to understand how to write the code to tune RoBERTa models used PyTorch,  and dealing with data.

## Notebooks explanation

- [**RoBERTa_tuning.ipynb**](https://github.com/SakshamGoelUK/COMP34812-NLU/blob/main/RoBERTa_tuning.ipynb): This is the main development notebook for the RoBERTa model. The preprocessing, hyperparameter selection, and training of the model is carried out in this notebook.
- [**RoBERTa_DEMO.ipynb**](https://github.com/SakshamGoelUK/COMP34812-NLU/blob/main/RoBERTa_DEMO.ipynb): This is the demo code that can be used to test the model, using csv inputs or individual sentence inputs. It was used to test on the development dataset throughout the development of the model, and was also used to generate the test dataset predictions.

## Running the Demo Code

#### Location of Fine-Tuned RoBERTa model: https://drive.google.com/drive/folders/17FtIWoRF8fjc0v9xHZGpzN2T1Fg4_UVJ?usp=sharing

Download the folder at this link, and follow the steps below. 
NOTE: The `model_path` must point to the folder, rather than the safetensors file, for it to be imported correctly.
To run the model and make predictions, ensure that you have the necessary paths configured for your environment:

1. Set `model_path` to the directory where the model and tokenizer are saved. (The model folder downloaded from the link above)
2. The `predict` function takes two strings - a claim and evidence - and outputs a binary prediction. You can use this function if you want to test a specific example and see the output.
3. To predict on a new CSV input file, set `link` to the path of your CSV file. The CSV should contain two columns: `Claim` and `Evidence`. The output will be a new CSV file with the predictions, with one column `prediction`.
4. Set `out` to the desired output location.

Make sure to update the path placeholders with the appropriate file paths for your specific setup.

## Software Requirements

- **Transformers**: 4.18.0
- **PyTorch**: 1.11.0+cu113



