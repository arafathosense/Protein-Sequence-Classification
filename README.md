# Protein Sequence Classification with Kolmogorov–Arnold Networks (KAN)

This project demonstrates **protein sequence classification** using a **Kolmogorov–Arnold Network (KAN)** on a **synthetic dataset**. The pipeline covers data generation, one-hot encoding of protein sequences, model definition, training, evaluation, and interpretability through symbolic regression.

The main objective is to show how KANs can learn **interpretable representations** from sequence-based biological data, offering an alternative to traditional black-box neural networks.


<img width="407" height="328" alt="image" src="https://github.com/user-attachments/assets/e0ad5d95-197d-4186-92d2-051668c2f108" />
<img width="407" height="328" alt="image" src="https://github.com/user-attachments/assets/f4c5cb5c-9a9a-401b-9d56-ec7b5ea45ce8" />

## 📌 Project Overview

Protein sequence classification is a fundamental task in bioinformatics, used in applications such as:

* Protein family classification
* Function prediction
* Motif and pattern recognition

In this project:

* Protein sequences are synthetically generated
* Amino acids are **one-hot encoded**
* A **Kolmogorov–Arnold Network (KAN)** is trained for classification
* Model performance is evaluated on unseen data
* Learned symbolic functions provide **interpretability**



## 🧠 Why Kolmogorov–Arnold Networks (KAN)?

KANs are inspired by the **Kolmogorov–Arnold representation theorem**, which states that any multivariate continuous function can be represented as a sum of univariate functions.

### Key advantages:

* ✅ Improved interpretability via symbolic regression
* ✅ Fewer parameters compared to dense neural networks
* ✅ Strong performance on structured and tabular data
* ✅ Transparent functional representations of learned features



## 🧬 Dataset Description

This project uses a **synthetic protein sequence dataset**.

### Dataset characteristics:

* **Alphabet**: 20 standard amino acids
* **Sequence length**: Fixed-length sequences
* **Encoding**: One-hot encoding
* **Labels**: Binary or multi-class (depending on configuration)

Synthetic data is used to:

* Control sequence patterns
* Clearly evaluate learning behavior
* Avoid dataset licensing restrictions


## ⚙️ Pipeline Overview

1. **Protein Sequence Generation**

   * Randomly generate amino acid sequences
   * Assign class labels based on predefined rules or motifs

2. **One-Hot Encoding**

   * Convert each amino acid into a 20-dimensional binary vector
   * Flatten encoded sequences for model input

3. **Dataset Preparation**

   * Split data into training and testing sets
   * Convert to tensors for model consumption

4. **KAN Model Definition**

   * Define KAN layers
   * Configure grid size and spline order

5. **Training**

   * Optimize using gradient-based methods
   * Minimize classification loss

6. **Evaluation**

   * Measure accuracy on test data
   * Analyze learned symbolic functions



## 🏗️ Model Architecture

The KAN model consists of:

* **Input layer**: One-hot encoded protein sequence vectors
* **KAN layers**: Learn univariate spline-based transformations
* **Output layer**: Classification logits

Unlike traditional MLPs, KANs replace linear weights with **learned functions**, enabling symbolic interpretation.



## 📈 Training Details

* **Loss Function**: Cross-entropy loss
* **Optimizer**: Adam or SGD
* **Epochs**: Configurable
* **Batch Size**: Configurable

Training converges by learning meaningful symbolic mappings between amino acid patterns and class labels.



## 📊 Evaluation Metrics

The model is evaluated using:

* **Classification Accuracy**
* **Loss Curves**

Optionally:

* Precision / Recall
* Confusion Matrix



## 🔍 Interpretability & Symbolic Regression

One of the key strengths of KANs is interpretability.

After training, the model can:

* Display learned univariate functions
* Reveal which amino acid positions influence predictions
* Approximate learned functions using symbolic expressions

This makes KANs particularly suitable for scientific and biological research.



## 🧪 Example Use Cases

* Educational demonstrations of interpretable ML in bioinformatics
* Rapid prototyping of sequence classification models
* Exploring symbolic learning on biological sequences


## 🛠️ Dependencies

* Python 3.8+
* NumPy
* PyTorch
* KAN library (if external implementation is used)


## 📌 Notes

* This project uses **synthetic data** and is not intended for direct biological inference
* Results may vary depending on random seed and dataset configuration


## 🤝 Acknowledgements

* Kolmogorov–Arnold Representation Theorem
* Research community on interpretable machine learning



## ✨ Future Work

* Apply KANs to real protein datasets (e.g., UniProt-derived tasks)
* Extend to variable-length sequences
* Compare performance with CNNs and Transformers
* Enhance symbolic extraction and visualization

## 👤 Author

**HOSEN ARAFAT**  

**Software Engineer, China**  

**GitHub:** https://github.com/arafathosense

**Researcher: Artificial Intelligence, Machine Learning, Deep Learning, Computer Vision, Image Processing**
