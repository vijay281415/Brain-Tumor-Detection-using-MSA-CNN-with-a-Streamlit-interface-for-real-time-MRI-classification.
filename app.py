import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, Model
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Detection - MSA-CNN",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .glioma {background-color: #ffebee;}
    .meningioma {background-color: #fff3e0;}
    .no-tumor {background-color: #e8f5e9;}
    .pituitary {background-color: #e3f2fd;}
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🧠 Brain Tumor Detection System (MSA-CNN)</h1>', unsafe_allow_html=True)
st.markdown("---")

# Model path
MODEL_PATH = r"C:\brain_tumor\model.keras"

# Define custom layers for MSA-CNN
class AttentionBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.filters = filters
    
    def build(self, input_shape):
        self.conv_query = layers.Conv2D(self.filters // 8, 1, padding='same')
        self.conv_key = layers.Conv2D(self.filters // 8, 1, padding='same')
        self.conv_value = layers.Conv2D(self.filters, 1, padding='same')
        self.conv_output = layers.Conv2D(self.filters, 1, padding='same')
        self.softmax = layers.Softmax(axis=-1)
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        
        query = self.conv_query(inputs)
        key = self.conv_key(inputs)
        value = self.conv_value(inputs)
        
        query = tf.reshape(query, [batch_size, height * width, self.filters // 8])
        key = tf.reshape(key, [batch_size, height * width, self.filters // 8])
        value = tf.reshape(value, [batch_size, height * width, self.filters])
        
        attention = tf.matmul(query, key, transpose_b=True)
        attention = self.softmax(attention)
        attended = tf.matmul(attention, value)
        attended = tf.reshape(attended, [batch_size, height, width, self.filters])
        
        output = self.conv_output(attended)
        return inputs + output
    
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config

class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
    
    def build(self, input_shape):
        self.conv1 = layers.Conv2D(self.filters, self.kernel_size, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(self.filters, self.kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        
        if input_shape[-1] != self.filters:
            self.projection = layers.Conv2D(self.filters, 1, padding='same')
            self.proj_bn = layers.BatchNormalization()
        else:
            self.projection = None
    
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        if self.projection:
            shortcut = self.projection(inputs)
            shortcut = self.proj_bn(shortcut, training=training)
        else:
            shortcut = inputs
        
        output = self.relu(x + shortcut)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "kernel_size": self.kernel_size})
        return config

class MultiScaleBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super(MultiScaleBlock, self).__init__(**kwargs)
        self.filters = filters
    
    def build(self, input_shape):
        self.conv1x1 = layers.Conv2D(self.filters // 4, 1, padding='same')
        self.conv3x3 = layers.Conv2D(self.filters // 4, 3, padding='same')
        self.conv5x5 = layers.Conv2D(self.filters // 4, 5, padding='same')
        self.maxpool = layers.MaxPooling2D(3, strides=1, padding='same')
        self.conv_pool = layers.Conv2D(self.filters // 4, 1, padding='same')
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
    
    def call(self, inputs, training=None):
        branch1 = self.conv1x1(inputs)
        branch2 = self.conv3x3(inputs)
        branch3 = self.conv5x5(inputs)
        branch4 = self.maxpool(inputs)
        branch4 = self.conv_pool(branch4)
        
        output = layers.concatenate([branch1, branch2, branch3, branch4], axis=-1)
        output = self.bn(output, training=training)
        output = self.relu(output)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config

@st.cache_resource
def load_model(model_path):
    """Load the trained MSA-CNN model with custom layers"""
    try:
        model = keras.models.load_model(
            model_path,
            custom_objects={
                'AttentionBlock': AttentionBlock,
                'ResidualBlock': ResidualBlock,
                'MultiScaleBlock': MultiScaleBlock
            }
        )
        return model, None
    except Exception as e:
        return None, str(e)

def preprocess_image(image):
    """Preprocess image for MSA-CNN model (168x168, grayscale, normalized)"""
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize to 168x168 (MSA-CNN input size)
    img_resized = cv2.resize(image, (168, 168), interpolation=cv2.INTER_LINEAR)
    
    # Normalize pixel values to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Add channel dimension and batch dimension
    img_processed = np.expand_dims(img_normalized, axis=-1)
    img_processed = np.expand_dims(img_processed, axis=0)
    
    return img_processed, img_resized

def predict_tumor(model, image):
    """Predict tumor type from MRI image"""
    # Class names from your training code
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    
    # Make prediction
    predictions = model.predict(image, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100
    
    return predictions[0], predicted_class, confidence, class_names

def create_probability_chart(predictions, class_names):
    """Create a bar chart of prediction probabilities"""
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#ef5350', '#ff9800', '#66bb6a', '#42a5f5']
    bars = ax.barh(class_names, predictions * 100, color=colors)
    ax.set_xlabel('Confidence (%)', fontsize=12)
    ax.set_title('Prediction Confidence for Each Class', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    
    # Add percentage labels
    for bar, pred in zip(bars, predictions):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{pred*100:.2f}%', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

# Sidebar
with st.sidebar:
    st.header("ℹ️ About MSA-CNN")
    st.info("""
    **Multi-Scale Attention CNN**
    
    Novel architecture combining:
    - 🔄 Multi-scale feature extraction
    - 👁️ Spatial attention mechanisms
    - 🔗 Residual connections
    
    **Classes Detected:**
    - 🔴 Glioma
    - 🟠 Meningioma  
    - 🟢 No Tumor
    - 🔵 Pituitary Tumor
    
    **Performance Metrics:**
    - Training Accuracy: 99.46%
    - Validation Accuracy: 98.55%
    - Test Accuracy: 98.55%
    - F1-Score: 98.47%
    """)
    
    st.header("📋 Instructions")
    st.markdown("""
    1. Upload an MRI brain scan (168×168 grayscale)
    2. Supported formats: JPG, PNG, JPEG
    3. Click 'Analyze with MSA-CNN'
    4. View predictions with confidence scores
    """)
    
    st.header("🏗️ Model Architecture")
    st.markdown("""
    **Total Parameters:** 9,020,900
    - Input: 168×168×1 grayscale
    - 4 stages with multi-scale blocks
    - Attention mechanisms at each stage
    - Global average pooling
    - Dense layers with dropout
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload MRI Image")
    uploaded_file = st.file_uploader(
        "Choose an MRI scan image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a brain MRI scan for tumor detection"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI Scan', use_container_width=True)

with col2:
    st.header("🔍 MSA-CNN Analysis Results")
    
    if uploaded_file is not None:
        # Load model
        with st.spinner('Loading MSA-CNN model...'):
            model, error = load_model(MODEL_PATH)
        
        if error:
            st.error(f"❌ Error loading model: {error}")
            st.info("Please ensure the model.keras file exists at the specified path.")
        else:
            st.success("✅ MSA-CNN model loaded successfully!")
            
            # Analyze button
            if st.button("🔬 Analyze with MSA-CNN", type="primary", use_container_width=True):
                with st.spinner('Analyzing image with MSA-CNN...'):
                    try:
                        # Preprocess image
                        processed_img, display_img = preprocess_image(image)
                        
                        # Make prediction
                        predictions, predicted_class, confidence, class_names = predict_tumor(model, processed_img)
                        
                        # Display results
                        st.markdown("### 🎯 Prediction Result")
                        
                        # Color-coded result box
                        class_colors = {
                            'Glioma': 'glioma',
                            'Meningioma': 'meningioma',
                            'No Tumor': 'no-tumor',
                            'Pituitary': 'pituitary'
                        }
                        
                        st.markdown(f"""
    <div class="prediction-box" style="background-color:white; color:black;">
        <h2 style="margin: 0; text-align: center;">{predicted_class}</h2>
        <h3 style="margin: 10px 0 0 0; text-align: center;">
            Confidence: {confidence:.2f}%
        </h3>
    </div>
""", unsafe_allow_html=True)

                        
                        # Display all probabilities
                        st.markdown("### 📊 Detailed Class Probabilities")
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.metric(
                                label="🔴 Glioma",
                                value=f"{predictions[0]*100:.2f}%"
                            )
                            st.metric(
                                label="🟠 Meningioma",
                                value=f"{predictions[1]*100:.2f}%"
                            )
                        
                        with col_b:
                            st.metric(
                                label="🟢 No Tumor",
                                value=f"{predictions[2]*100:.2f}%"
                            )
                            st.metric(
                                label="🔵 Pituitary",
                                value=f"{predictions[3]*100:.2f}%"
                            )
                        
                        # Visualization
                        st.markdown("### 📈 Confidence Distribution")
                        fig = create_probability_chart(predictions, class_names)
                        st.pyplot(fig)
                        
                        # Additional info based on prediction
                        st.markdown("### 📝 Clinical Information")
                        if predicted_class == "Glioma":
                            st.warning("""
                            **Glioma Detected**
                            - Most common type of primary brain tumor
                            - Arises from glial cells (supportive brain tissue)
                            - Can be low-grade or high-grade (WHO I-IV)
                            - Infiltrative growth pattern
                            - **Action Required:** Immediate neurological consultation
                            - May require biopsy for grading
                            """)
                        elif predicted_class == "Meningioma":
                            st.warning("""
                            **Meningioma Detected**
                            - Usually benign (90% of cases)
                            - Arises from meninges (brain coverings)
                            - Typically slow-growing with well-defined borders
                            - Extra-axial location (outside brain tissue)
                            - **Action Required:** Consult neurosurgeon
                            - Treatment options: observation, surgery, or radiation
                            """)
                        elif predicted_class == "No Tumor":
                            st.success("""
                            **No Tumor Detected**
                            - Normal brain tissue patterns identified
                            - No signs of mass lesions
                            - Regular follow-up as per clinical indication
                            - Continue routine health monitoring
                            - If symptoms persist, consult healthcare provider
                            """)
                        else:  # Pituitary
                            st.warning("""
                            **Pituitary Tumor Detected**
                            - Located in pituitary gland (sella turcica region)
                            - Can be functional (hormone-secreting) or non-functional
                            - Usually benign adenomas
                            - May cause hormonal imbalances or vision problems
                            - **Action Required:** Consult endocrinologist AND neurosurgeon
                            - Hormonal testing recommended
                            """)
                        
                        # Performance metrics
                        st.markdown("### 📊 Model Performance Metrics")
                        perf_col1, perf_col2, perf_col3 = st.columns(3)
                        
                        with perf_col1:
                            st.metric("Training Accuracy", "99.46%")
                        with perf_col2:
                            st.metric("Validation Accuracy", "98.55%")
                        with perf_col3:
                            st.metric("Test Accuracy", "98.55%")
                        
                        # Class-specific metrics
                        st.markdown("### 🎯 Per-Class Performance")
                        metrics_data = {
                            "Class": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
                            "Precision": ["98.34%", "99.32%", "99.02%", "97.39%"],
                            "Recall": ["98.67%", "95.42%", "100.00%", "99.67%"],
                            "F1-Score": ["98.50%", "97.33%", "99.51%", "98.52%"]
                        }
                        st.table(metrics_data)
                        
                        st.info("⚠️ **Medical Disclaimer:** This is an AI-assisted diagnostic support tool using the MSA-CNN architecture. It achieves 98.55% accuracy but should NOT replace professional medical diagnosis. Always consult qualified healthcare professionals (radiologists, neurologists, neurosurgeons) for final diagnosis and treatment decisions.")
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
    else:
        st.info("👆 Please upload an MRI image to begin MSA-CNN analysis")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🧠 <strong>Brain Tumor Detection System using MSA-CNN</strong></p>
    <p><strong>Multi-Scale Attention Convolutional Neural Network</strong></p>
    <p>Model: 9.02M parameters | Accuracy: 98.55% | F1-Score: 98.47%</p>
    <p>Architecture: Multi-Scale Blocks + Attention Mechanisms + Residual Connections</p>
    <p>Powered by TensorFlow 2.20.0 & Streamlit</p>
</div>
""", unsafe_allow_html=True)