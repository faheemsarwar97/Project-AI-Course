# Agri-Feed: AI-Powered Smart Leaf Diagnosis and Prevention System

An AI-powered MVP system for plant leaf disease diagnosis using PyTorch and treatment recommendations using LLM integration.

## Project Overview

Agri-Feed is a complete MVP implementation that combines:
- **Computer Vision (CV)**: PyTorch-based deep learning model using ResNet18 for leaf disease classification
- **Large Language Model (LLM)**: OpenAI GPT-4 integration for generating contextual treatment recommendations

The system addresses the challenge of plant disease management by providing:
1. Automated visual diagnosis of leaf diseases
2. Confidence-scored predictions
3. AI-generated treatment recommendations based on diagnosis, location, and climate context

## Project Structure

```
Faheem/
├── notebooks/
│   └── agri_feed_mvp.ipynb          # Main Jupyter notebook
├── models/
│   └── .gitkeep                      # Placeholder for saved models
├── data/
│   └── .gitkeep                      # Placeholder for sample images
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

## Features

- **Image Upload**: Interactive file upload widget for leaf images
- **Disease Classification**: Deep learning model for identifying 10 common plant diseases
- **Visualization**: Side-by-side display of uploaded image and prediction confidence scores
- **Treatment Recommendations**: AI-generated, context-aware treatment suggestions
- **Contextual Input**: Support for plant type, location, and climate information

## Setup Instructions

### Prerequisites

- **macOS** (optimized for macOS, including Apple Silicon M1/M2/M3)
- Python 3.8 or higher
- Anaconda or Miniconda (recommended for macOS)
- Jupyter Notebook or JupyterLab
- OpenAI API key (for LLM recommendations)

### Installation (macOS)

1. **Clone or navigate to the project directory:**
   ```bash
   cd Faheem
   ```

2. **Create a conda environment (recommended for macOS):**
   ```bash
   conda create -n agri-feed python=3.10 -y
   conda activate agri-feed
   ```
   
   **Note:** The notebook will automatically create this environment when you run the setup cells.

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   **For Apple Silicon (M1/M2/M3) Macs:**
   - PyTorch will automatically install with MPS (Metal Performance Shaders) support
   - This enables GPU acceleration on Apple Silicon Macs
   - The notebook will detect and use MPS automatically

4. **Set up OpenAI API key:**
   
   **Option 1: Environment Variable (Recommended)**
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```
   
   **Option 2: In the Notebook**
   ```python
   import os
   os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
   ```

5. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

6. **Open the notebook:**
   Navigate to `notebooks/agri_feed_mvp.ipynb` and run all cells.

## Usage Guide

### Basic Usage

1. **Run Setup Cells**: Execute the first few cells to import libraries and initialize the model

2. **Set API Key** (if not set as environment variable):
   ```python
   import os
   os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
   ```

3. **Use the Interactive Interface**:
   - Upload a leaf image using the file upload widget
   - (Optional) Enter plant type, location, and climate information
   - Click "🔍 Analyze Leaf" button
   - View diagnosis results and treatment recommendations

### Supported Disease Classes

The model currently supports classification of the following conditions:
- Healthy
- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Target Spot
- Tomato Mosaic Virus
- Yellow Leaf Curl Virus

### Model Information

- **Architecture**: ResNet18 with transfer learning
- **Input Size**: 224x224 pixels
- **Preprocessing**: ImageNet normalization
- **Output**: 10-class classification with confidence scores

**Note**: The model uses random weights by default. For production use:
- Train the model on a plant disease dataset (e.g., Plant Village dataset)
- Save trained weights to `models/leaf_disease_model.pth`
- The notebook will automatically load pre-trained weights if available

## API Key Configuration

### Getting an OpenAI API Key

1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in to your account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key and set it as described in Setup Instructions

### Security Best Practices

- **Never commit API keys to version control**
- Use environment variables for API keys
- Consider using `.env` files with `python-dotenv` for local development
- Rotate API keys regularly

## Dependencies

- `torch>=2.0.0` - PyTorch deep learning framework (with MPS support for Apple Silicon)
- `torchvision>=0.15.0` - Pre-trained models and image utilities
- `pillow>=10.0.0` - Image processing
- `matplotlib>=3.7.0` - Visualization
- `numpy>=1.24.0` - Numerical operations
- `ipywidgets>=8.0.0` - Interactive widgets
- `openai>=1.0.0` - OpenAI API client
- `jupyter>=1.0.0` - Jupyter notebook environment
- `ipython>=8.0.0` - Enhanced Python shell

## Troubleshooting

### Common Issues

1. **GPU/Acceleration not available**
   - **Apple Silicon (M1/M2/M3) Macs:** The notebook will automatically use MPS (Metal Performance Shaders) for GPU acceleration if available
   - **Intel Macs:** Will use CPU (CUDA is not available on macOS)
   - **Other systems:** Will use CUDA if available, otherwise CPU
   - The model will work on CPU but will be slower

2. **API Key Error**
   - Ensure `OPENAI_API_KEY` environment variable is set
   - Verify the API key is valid and has sufficient credits
   - Check internet connection

3. **Model Loading Issues**
   - If pre-trained weights are not found, the model uses random initialization
   - This is expected for the MVP - predictions will not be accurate without training

4. **Image Upload Not Working**
   - Ensure `ipywidgets` is properly installed
   - Try refreshing the notebook kernel
   - Check that the image file is a valid image format (JPG, PNG, etc.)

## Future Enhancements

- Model training pipeline with Plant Village dataset
- Support for additional plant types
- Batch processing of multiple images
- Export functionality for diagnosis reports
- Integration with local LLM models (e.g., Ollama)
- Mobile app deployment

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- Plant Village dataset for disease classification research
- PyTorch team for the deep learning framework
- OpenAI for LLM API access

