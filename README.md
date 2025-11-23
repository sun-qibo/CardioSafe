# Heart Failure Predictor Model

AutoGluon tabular model for heart failure prediction.

## Model Information

- **Framework**: AutoGluon Tabular
- **Task**: Binary Classification (Heart Failure Prediction)
- **Model Location**: [Hugging Face Hub](https://huggingface.co/qiiiibeau/heart-failure-predictor-model)

## Installation
```bash
pip install autogluon.tabular
pip install huggingface_hub
```

## Loading the Model

### Option 1: Load from Hugging Face Hub (Recommended)
```python
from huggingface_hub import snapshot_download
from autogluon.tabular import TabularPredictor

# Download the model from Hugging Face
model_path = snapshot_download(
    repo_id="qiiiibeau/heart-failure-predictor-model",
    repo_type="model"
)

# Load the predictor
predictor = TabularPredictor.load(model_path, require_py_version_match=False)

print("Model loaded successfully!")
```

### Option 2: Load from Local Directory

If you have the model files locally (e.g., downloaded zip):
```python
from autogluon.tabular import TabularPredictor

# Load from local path
predictor = TabularPredictor.load('./heart_failure_predictor_model')
```

## Making Predictions

### Single Prediction
```python
# Get prediction
prediction = predictor.predict(sample_data)
print(f"AUROC: {prediction[0]}")

# Get prediction probabilities
probabilities = predictor.predict_proba(sample_data)
print(f"Probabilities: {probabilities}")
```

## Example Usage in Gradio (for Hugging Face Spaces)
```python
import gradio as gr
from huggingface_hub import snapshot_download
from autogluon.tabular import TabularPredictor
import pandas as pd

# Load model
model_path = snapshot_download(repo_id="qiiiibeau/heart-failure-predictor-model")
predictor = TabularPredictor.load(model_path)

def predict(**kwargs):
    """Make prediction from input features"""
    # Convert inputs to DataFrame
    input_df = pd.DataFrame([kwargs])
    
    # Get predictions
    prediction = predictor.predict(input_df)[0]
    probabilities = predictor.predict_proba(input_df).iloc[0].to_dict()
    
    return {
        "prediction": str(prediction),
        "probabilities": probabilities
    }

# Create Gradio interface
# Update inputs based on your actual features
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="gene1"),
        gr.Number(label="gene2"),
        gr.Number(label="gene3"),
        # Add all features here
    ],
    outputs=gr.JSON(label="Prediction Results"),
    title="Heart Failure Predictor",
    description="Predict heart failure risk based on clinical features"
)

if __name__ == "__main__":
    demo.launch()
```


## Requirements

Create a `requirements.txt` for your deployment:
```
autogluon.tabular>=1.0.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
gradio>=4.0.0  # If using Gradio
```


## File Structure
```
heart_failure_predictor_model/
├── predictor.pkl          # Main predictor object
├── utils/                 # Utility files
├── models/               # Trained model files
│   ├── WeightedEnsemble_L2/
│   └── ...
└── ...
```

## Contact

For questions about the model or deployment, contact: sunqibo1210@gmail.com

## License

**TODO**: Add license information
