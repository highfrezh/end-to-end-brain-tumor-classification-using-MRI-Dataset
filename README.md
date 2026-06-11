# Brain Tumor Classification using MRI Scans

**End-to-End Deep Learning Project** for multi-class brain tumor detection from MRI images using **Transfer Learning (VGG16)**.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

---

## 📋 Project Overview

This project implements a complete **end-to-end CNN-based system** to classify brain tumors into four categories:
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

The model leverages **VGG16 transfer learning** and is built with a modular MLOps pipeline, including training, evaluation, and web deployment.

### 🎯 Key Achievements
- **94% Classification Accuracy** on the test set
- Implemented proper data preprocessing and augmentation
- Used Transfer Learning with VGG16 for high performance with limited data
- Containerized with Docker + CI/CD pipeline on GitHub Actions
- Deployed as a web application using Flask

---

## 🏗️ Project Structure

```
├── research/                  # Experimentation notebooks
├── src/cnnClassifier/         # Main source code
├── templates/                 # Frontend templates
├── config/                    # Configuration files
├── app.py                     # Flask web application
├── Dockerfile
├── requirements.txt
└── params.yaml
```

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/highfrezh/end-to-end-brain-tumor-classification-using-MRI-Dataset.git
cd end-to-end-brain-tumor-classification-using-MRI-Dataset
```

### 2. Create and activate conda environment
```bash
conda create -n cnn_env python=3.11 -y
conda activate cnn_env
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
python app.py
```

Then open your browser and go to `http://localhost:8080`

---

## 🐳 Docker Deployment

```bash
docker build -t brain-tumor-app .
docker run -p 8080:8080 brain-tumor-app
```

---

## 📊 Results

- **Test Accuracy**: **94%**
- Model: VGG16 (Transfer Learning)
- Techniques used: Data Augmentation, Fine-tuning, Dropout

---

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow, Keras, VGG16
- **Backend**: Flask
- **MLOps**: DVC, MLflow (if used), GitHub Actions
- **Deployment**: Docker, AWS EC2 + ECR
- **Visualization**: Matplotlib, Seaborn

---

## 📌 Future Improvements

- Experiment with newer architectures (EfficientNet, ResNet50, Vision Transformers)
- Implement Grad-CAM for model explainability
- Create a more interactive Streamlit/Gradio interface
- Deploy on Hugging Face Spaces or Render

---

## 👨‍💻 Author

**Ibraheem Olabintan**  
Computer Science Graduate | AI/ML Engineer

---

⭐ **Star this repository** if you found it helpful!

```
