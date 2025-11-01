# Plant Disease Detection System 🌱

A web-based AI-powered application for detecting plant diseases from leaf images. This system uses machine learning to identify various plant diseases in tomatoes, potatoes, and peppers, providing accurate diagnosis and treatment recommendations.

## Features

- **AI-Powered Detection**: Utilizes TensorFlow.js with a custom trained model for disease classification
- **Multi-Plant Support**: Supports detection for Tomato, Potato, and Pepper plants
- **Comprehensive Diagnosis**: Provides detailed information including symptoms, treatment, and prevention methods
- **User-Friendly Interface**: Drag-and-drop image upload with intuitive web interface
- **Real-time Analysis**: Instant results with confidence scores
- **Responsive Design**: Works on desktop and mobile devices

## Technologies Used

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **AI/ML**: TensorFlow.js
- **Styling**: Custom CSS with Google Fonts (Poppins, Quicksand)
- **Model Format**: TensorFlow.js GraphModel

## Installation

1. **Clone or Download** the project files to your local machine
2. **Ensure Model Files**: Make sure the `tfjs_model/` folder and `class_names.json` are present in the project root
3. **Open in Browser**: Simply open `index.html` in any modern web browser

No additional installation or server setup required - it's a client-side application!

## Model Training

The AI model used in this application was trained using the provided Python script on Google Colab. To train your own model:

### Prerequisites
- Google Colab account
- Kaggle API key (`kaggle.json`)

### Training Steps
1. Upload `training-model/main.py` to Google Colab
2. Upload your `kaggle.json` file when prompted by the script
3. Run all cells in the notebook
4. The script will:
   - Download the plant disease dataset from Kaggle
   - Preprocess the data
   - Train a MobileNetV2-based CNN model
   - Convert the trained model to TensorFlow.js format
   - Download the converted model files

### Files Required for Training
- `training-model/main.py` - Main training script
- `kaggle.json` - Kaggle API credentials for dataset download

**Note**: The Python training code is designed to run on Google Colab for optimal performance and GPU acceleration.

## Usage

1. **Launch the Application**: Open `index.html` in your web browser
2. **Wait for Model Loading**: The AI model will load automatically (may take a few seconds)
3. **Upload Image**: Click the upload area or drag & drop a leaf image (JPG/PNG, max 5MB)
4. **Analyze**: Click "🔍 Analisa Penyakit" to process the image
5. **View Results**: Get detailed diagnosis with treatment recommendations

## Supported Diseases

The system can detect the following plant diseases:

### Tomato Diseases
- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites (Two-spotted)
- Target Spot
- Yellow Leaf Curl Virus
- Tomato Mosaic Virus
- Healthy

### Potato Diseases
- Early Blight
- Late Blight
- Healthy

### Pepper Diseases
- Bacterial Spot
- Healthy

## Model Details

- **Architecture**: Custom trained convolutional neural network
- **Input Size**: 224x224 pixels
- **Classes**: 15 disease categories across 3 plant types
- **Accuracy**: High accuracy on trained dataset (results may vary with real-world images)

## Project Structure

```
plant-disease-detection/
├── index.html          # Main HTML page
├── style.css           # CSS styling
├── script.js           # JavaScript logic and AI integration
├── class_names.json    # Model class labels
├── tfjs_model/         # TensorFlow.js model files
│   ├── model.json
│   ├── group1-shard1of3.bin
│   ├── group1-shard2of3.bin
│   └── group1-shard3of3.bin
├── training-model/
│   └── main.py         # Python training script for Google Colab
└── README.md           # This file
```

## Browser Compatibility

- Chrome 88+
- Firefox 85+
- Safari 14+
- Edge 88+

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is open source and available under the [MIT License](LICENSE).

## Disclaimer

This application is for educational and informational purposes only. Always consult with agricultural experts for professional plant disease diagnosis and treatment recommendations.

---

**Built with ❤️ for sustainable agriculture**
