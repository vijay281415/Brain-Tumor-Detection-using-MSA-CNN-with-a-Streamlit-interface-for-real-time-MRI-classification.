# Brain-Tumor-Detection-using-MSA-CNN-with-a-Streamlit-interface-for-real-time-MRI-classification.
Brain Tumor Detection System using Multi-Scale Attention CNN (MSA-CNN) for accurate MRI image classification. Built with Streamlit, it provides real-time predictions, confidence scores, and visualization, supporting detection of Glioma, Meningioma, Pituitary, and No Tumor cases.  Option 2:
# 🧠 Brain Tumor Detection System (MSA-CNN)

A deep learning-based web application for detecting and classifying brain tumors from MRI images using a **Multi-Scale Attention Convolutional Neural Network (MSA-CNN)**.

---

## 🚀 Features

* 🔍 Upload MRI brain scan images
* 🧠 Detect tumor types:

  * Glioma
  * Meningioma
  * Pituitary Tumor
  * No Tumor
* 📊 Confidence score visualization
* 📈 Probability distribution chart
* 🧾 Clinical insights for each prediction
* 🌐 Interactive UI built with Streamlit

---

## 🏗️ Model Architecture

The system uses a custom deep learning model combining:

* 🔄 Multi-scale feature extraction
* 👁️ Attention mechanisms
* 🔗 Residual connections

**Model Details:**

* Input size: 168 × 168 (grayscale)
* Parameters: ~9M
* Framework: TensorFlow / Keras

---

## 📊 Performance

* ✅ Training Accuracy: 99.46%
* ✅ Validation Accuracy: 98.55%
* ✅ Test Accuracy: 98.55%
* ✅ F1 Score: 98.47%

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Streamlit
* Matplotlib

---

## 📂 Project Structure

```
├── app.py                 # Main Streamlit application
├── model.keras           # Trained model
├── requirements.txt      # Dependencies
└── README.md             # Documentation
```

---

## ⚙️ Installation & Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

---

## 📌 Usage

1. Upload an MRI image (JPG/PNG)
2. Click **Analyze**
3. View:

   * Predicted tumor type
   * Confidence score
   * Detailed probability chart

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.
It should **not be used for medical diagnosis** without professional consultation.

---

## 📸 Demo

(Add screenshots here)

---

## 🤝 Contributing

Feel free to fork and improve this project!

---

## 📜 License

MIT License
