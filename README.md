# Sentiment Analysis with DistilBERT

This project implements a sentiment analysis model using **DistilBERT** from Hugging Face Transformers, fine-tuned on the **SST-2 dataset**. It includes scripts for both training (with evaluation) and inference.

---

## 🚀 Features

* Fine-tunes `distilbert-base-uncased` for binary sentiment classification
* Uses SST-2 dataset from Hugging Face Datasets
* Training with Hugging Face `Trainer` API
* Evaluation during training
* Separate inference pipeline for predictions
* Easily runnable on Google Colab

---

## 🧰 Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Scikit-learn

---

## 📦 Installation

Clone the repository:

```
git clone https://github.com/pawansardar/sentiment-analysis.git
cd sentiment-analysis
```

Install dependencies:

```
pip install -r requirements.txt
```

> ⚠️ For Google Colab, it is recommended to enable GPU:
> Runtime → Change runtime type → GPU

---

## 🏋️ Training

Run the training script:

```
python -m scripts.train
```

This will:

* Load the SST-2 dataset
* Tokenize text using DistilBERT tokenizer
* Fine-tune the model
* Evaluate on the validation set
* Save training logs and checkpoints

> * Training logs, checkpoints, and intermediate results are saved in `./experiments`
> * The best trained model is saved in `./final_model`

---

## 🔍 Inference

Run the inference script:

```
python -m scripts.predict
```

> ✏️ The input text is currently defined inside the script.
> To test different inputs, modify the text in `scripts/predict.py` and rerun the script.

Example:

```
Input: "This movie was amazing!"
Output: { label: Positive, confidence: 0.98 }
```

---

## ☁️ Run on Google Colab

1. Open Google Colab

2. Clone the repository:

```
!git clone https://github.com/pawansardar/sentiment-analysis.git
%cd sentiment-analysis
```

3. Install dependencies:

```
!pip install -r requirements.txt
```

4. Run training:

```
!python -m scripts.train
```

---

## 🧠 Model Details

* Base Model: `distilbert-base-uncased`
* Task: Binary Sentiment Classification
* Dataset: SST-2
* Framework: PyTorch + Hugging Face Trainer

---

## 📊 Evaluation Metrics

The model is evaluated using the following metrics:

* **Accuracy** – Measures overall correctness of predictions
* **F1 Score** – Balances precision and recall, useful for classification tasks

These metrics are computed using `scikit-learn` during evaluation.

---

## 🔮 Future Improvements

* Add CLI support for inference inputs
* Add support for multi-class sentiment classification
* Deploy model as a REST API
* Add visualization for training metrics
* Hyperparameter tuning

---

## 📜 License

This project is open-source and available under the MIT License.
