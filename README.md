# 🩺 Skin Disease Classification using Deep Learning

A Deep Learning-based web application that classifies dermoscopic skin lesion images into seven different skin disease categories using a fine-tuned **ResNet50** model. The project includes the complete deep learning pipeline—from data preprocessing and model experimentation to deployment with **Streamlit** for real-time inference.

---

# 📖 Project Overview

Early detection of skin diseases can significantly improve treatment outcomes. However, manual diagnosis from dermoscopic images requires clinical expertise and can be time-consuming.

This project aims to assist in skin disease classification by leveraging **Transfer Learning** with **ResNet50**. Multiple deep learning models were trained and evaluated on the **HAM10000** dataset before selecting the final deployment model based on overall performance.

The application allows users to upload a skin lesion image and receive:

* Predicted skin disease
* Prediction confidence
* Top-3 predicted classes
* Probability distribution across all classes
* Brief description of the predicted disease

The complete system has been deployed as a user-friendly **Streamlit** web application.

---

# ✨ Features

* 📷 Upload dermoscopic skin lesion images
* 🤖 Automatic disease prediction using Deep Learning
* 🎯 Confidence score for every prediction
* 🏆 Top-3 predicted disease classes
* 📊 Prediction probability visualization
* 📋 Disease description
* 🌐 Interactive Streamlit web interface
* ⚠️ Medical disclaimer for responsible usage

---

# 🏗️ Project Pipeline

```text
                 HAM10000 Dataset
                        │
                        ▼
              Data Preprocessing
                        │
                        ▼
             Image Normalization
                        │
                        ▼
          Deep Learning Model Training
        ┌──────────┬──────────────┬──────────────┐
        │          │              │              │
        ▼          ▼              ▼
   Custom CNN  EfficientNetB0  ResNet50
        │          │              │
        └──────────┴──────────────┘
                    │
                    ▼
            Model Evaluation
                    │
                    ▼
         Best Model Selection
                    │
                    ▼
          Streamlit Deployment
```

---

# 🧠 Model Development

Rather than directly selecting a pretrained model, multiple deep learning architectures were trained and evaluated to identify the most suitable model for skin disease classification.

The development process followed a progressive experimentation approach:

### Phase 1 — Baseline Model

A custom Convolutional Neural Network (CNN) was developed from scratch to establish a performance baseline.

The purpose of this model was to understand the dataset characteristics and evaluate how a manually designed CNN performs without transfer learning.

---

### Phase 2 — Transfer Learning with EfficientNetB0

EfficientNetB0 was then used as a pretrained feature extractor.

The feature extraction approach significantly improved the classification performance compared to the baseline CNN.

An additional fine-tuning experiment was performed by unfreezing part of the network. However, this configuration resulted in degraded validation performance and reduced generalization. Therefore, the fine-tuned EfficientNet model was not selected for deployment.

---

### Phase 3 — Transfer Learning with ResNet50

Finally, ResNet50 was trained using transfer learning and later fine-tuned.

After evaluating all trained models, ResNet50 demonstrated the best balance between classification accuracy, validation performance, and overall stability.

For this reason, the fine-tuned ResNet50 model was selected as the final deployment model.

---

# 📊 Model Comparison

| Model          | Training Strategy                   |                    Test Accuracy | Decision         |
| -------------- | ----------------------------------- | -------------------------------: | ---------------- |
| Custom CNN     | Built From Scratch                  |                           60.55% | Baseline Model   |
| EfficientNetB0 | Feature Extraction                  |                           72.06% | Good Performance |
| EfficientNetB0 | Fine-Tuning                         | Validation performance decreased | Rejected         |
| **ResNet50**   | **Transfer Learning + Fine-Tuning** |       **Final Deployment Model** | ✅ Selected       |

---

# 🎯 Why ResNet50?

Although EfficientNetB0 achieved competitive performance during feature extraction, the fine-tuned ResNet50 model provided better overall stability during experimentation.

The final model was selected based on multiple considerations rather than a single accuracy metric:

* Better overall generalization
* Stable validation performance
* Reliable prediction behaviour
* Better suitability for deployment

Instead of choosing the model with only the highest training accuracy, the final decision prioritized real-world inference performance and deployment stability.

---

# 📈 Model Performance

### Final Selected Model

**Architecture:** ResNet50 (Fine-Tuned)

### Training Results

| Metric              |                    Value |
| ------------------- | -----------------------: |
| Training Accuracy   | *(Add your final value)* |
| Validation Accuracy | *(Add your final value)* |
| Test Accuracy       | *(Add your final value)* |

---

# 🔍 Inference Pipeline

The deployed application performs the following steps during prediction:

1. User uploads a dermoscopic skin lesion image.
2. The image is resized to **224 × 224** pixels.
3. ResNet50 preprocessing is applied.
4. The fine-tuned ResNet50 model generates prediction probabilities.
5. The class with the highest probability is selected.
6. The application displays:

   * Predicted disease
   * Confidence score
   * Top-3 predictions
   * Probability distribution chart
   * Disease description

---

# 💡 Challenges Faced

During development, several practical challenges were encountered:

* Class imbalance in the HAM10000 dataset.
* Lower performance of the initial custom CNN.
* Performance degradation during EfficientNetB0 fine-tuning.
* Hyperparameter experimentation for transfer learning.
* Building a modular inference pipeline for deployment.
* Integrating the trained model into a Streamlit application.

Addressing these challenges helped improve both the final model and the deployment workflow.


# 📂 Dataset

**Dataset Name:** HAM10000 (Human Against Machine with 10,000 Training Images)

The HAM10000 dataset is a publicly available collection of dermoscopic skin lesion images widely used for skin disease classification research.

### Number of Classes

| Class Code | Disease Name         |
| ---------- | -------------------- |
| akiec      | Actinic Keratoses    |
| bcc        | Basal Cell Carcinoma |
| bkl        | Benign Keratosis     |
| df         | Dermatofibroma       |
| mel        | Melanoma             |
| nv         | Melanocytic Nevus    |
| vasc       | Vascular Lesion      |

---

# 📁 Project Structure

```text
SkinDiseaseClassifier/

│── app.py                     # Streamlit web application
│── inference.py               # Model inference pipeline
│── requirements.txt           # Project dependencies
│── README.md                  # Project documentation
│── .gitignore

├── models/
│     └── resnet50_finetuned_model.keras

├── utils/
│     ├── preprocess.py
│     └── mapping.py

├── sample_images/
│     ├── sample1.jpg
│     ├── sample2.jpg
│     └── sample3.jpg

├── results/
│     ├── app_demo.png
│     ├── confusion_matrix.png
│     ├── model_comparison.png
│     └── probability_chart.png

└── assets/
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/pal-deepaak/Skin_Cancer_Classifier.git
```

Move into the project directory

```bash
cd SkinDiseaseClassifier
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the virtual environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install all dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit application

```bash
streamlit run app.py
```

The application will automatically open in your default browser.

---

# 📷 Application Screenshots

## Home Page

![Home Page](result/uploadImg.png)

Example:

```text
results/app_demo.png
```

---

## Prediction Output
![Prediction Output](result/prediction.png)

---

## Probability Distribution

*(Insert screenshot here)*

---

## Model Comparison

Test Accuracy

Custom CNN         ████████████ 60.55%

EfficientNetB0     ████████████████ 72.06%

ResNet50           ██████████████████ 80.50

---

# 🧪 Sample Prediction Workflow

1. Upload a dermoscopic skin lesion image.
2. Click **Predict Disease**.
3. The AI model analyzes the uploaded image.
4. The predicted disease is displayed.
5. Confidence score is shown.
6. Top-3 predictions are displayed.
7. Probability distribution chart is generated.
8. Disease description is displayed.
9. Read the medical disclaimer.

🚀 Future Improvements

This project can be further enhanced with several advanced features:

Grad-CAM visualization for explainable AI
Multi-model ensemble learning
Confidence calibration
Docker containerization
Cloud deployment (AWS / Azure / GCP)
REST API using FastAPI
User authentication
PDF report generation
Multi-language support

⚠️ Medical Disclaimer

This project is developed solely for educational and research purposes.

The predictions generated by this application should NOT be considered a substitute for professional medical diagnosis or treatment.

Always consult a qualified dermatologist or healthcare professional before making any medical decisions.

🤝 Acknowledgements

Special thanks to:

TensorFlow
Keras
Streamlit
OpenCV
NumPy
Matplotlib
HAM10000 Dataset Authors

Their excellent open-source contributions made this project possible.

📚 References
HAM10000 Dataset
TensorFlow Documentation
Keras Documentation
Streamlit Documentation
ResNet50 Research Paper
👨‍💻 Developer

Deepak Kumar

BCA Graduate

Aspiring AI / Machine Learning Engineer

⭐ Support

If you found this project helpful, consider giving this repository a ⭐ on GitHub.#   S k i n _ C a n c e r _ C l a s s i f i e r  
 