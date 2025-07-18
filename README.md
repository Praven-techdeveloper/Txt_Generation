Creative Text Generator with Streamlit
Overview
This Streamlit application leverages Hugging Face Transformers to generate creative text formats including poetry, code snippets, movie scripts, and song lyrics. Choose between lightweight GPT-2 for quick experiments or powerful GPT-J-6B for high-quality creative output.

Features
🎭 Multiple creative formats: Generate poetry, code, scripts, and song lyrics

🤖 Model selection: Choose between GPT-2 (fast) and GPT-J-6B (high quality)

🎚️ Parameter control: Adjust creativity, length, and diversity settings

⚡ Optimized performance: 8-bit quantization for efficient GPU usage

🎨 Smart formatting: Automatic code highlighting and script formatting

💾 Model caching: Loaded models stay cached for faster subsequent runs

Requirements
Hardware
Model	RAM	GPU VRAM	Recommendation
GPT-2	2GB+	Not required	Any computer
GPT-J-6B	12GB+	8GB+	NVIDIA RTX 3080+
Software
Python 3.8+

CUDA Toolkit 11.7+ (for GPU acceleration)

NVIDIA drivers (for GPU acceleration)

Installation
1. Create and activate virtual environment
bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
2. Install dependencies
bash
pip install -r requirements.txt
3. Set up environment variables
Create a .env file in the project root:

env
# For Hugging Face Hub access (optional)
HUGGINGFACE_TOKEN=your_token_here
Usage
Running the application
bash
streamlit run app.py
The application will open in your default browser at http://localhost:8501

Command-line options
bash
# Run on a different port
streamlit run --server.port 8502 app.py

# Enable auto-reload during development
streamlit run --server.runOnSave true app.py

# Make accessible on local network
streamlit run --server.address 0.0.0.0 app.py
Application Interface
Sidebar Controls:

Select model (GPT-2 or GPT-J-6B)

Adjust max length (50-500 tokens)

Set temperature (0.1-1.0)

Configure Top-K diversity (1-100)

Set random seed for reproducibility

Main Interface:

Enter your creative prompt in the text area

Use preset buttons for quick examples

Click "Generate Text" to create content

Output will be formatted based on content type

Deployment Options
Docker Deployment
Build the Docker image:

bash
docker build -t text-generator .
Run the container:

bash
docker run -p 8501:8501 text-generator
Cloud Deployment Platforms
Streamlit Sharing:

Push to GitHub repository

Connect repository at share.streamlit.io

Set Python version to 3.8+

Google Cloud Run:

bash
gcloud run deploy text-generator --source . --port 8501
AWS App Runner:

Create new service from container registry

Set port to 8501

Troubleshooting
Common Issues
Out of Memory Error:

Reduce max_length parameter

Use GPT-2 instead of GPT-J

Add GPU with more VRAM

Model Loading Failure:

Update packages: pip install --upgrade transformers accelerate bitsandbytes

Clear cache: rm -r ~/.cache/huggingface

CUDA Compatibility Issues:

Verify CUDA installation: nvcc --version

Ensure torch supports your CUDA version:

bash
pip uninstall torch
pip install torch --extra-index-url https://download.pytorch.org/whl/cu117
Performance Optimization
For better performance with GPT-J-6B:

python
# Use 4-bit quantization (requires transformers >= 4.30)
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

generator = pipeline(
    "text-generation",
    model="EleutherAI/gpt-j-6B",
    quantization_config=quant_config
)
 

