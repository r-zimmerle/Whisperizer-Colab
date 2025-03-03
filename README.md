# Automatic Audio Transcription with Whisper

This repository contains a **Google Colab notebook** for automatically transcribing audio extracted from videos using OpenAI's **Whisper** model. The notebook also checks for GPU availability to accelerate processing.

## 📌 Project Features

- **Convert Videos to Audio** 🎥🔊  
    Extracts audio from `.mp4` files using FFmpeg (supports `.mov`, `.avi`, etc.).
- **Automatic Transcription** 📝  
    Uses OpenAI's **Whisper 'turbo'** model for fast and accurate transcription.
- **Runs on Google Colab** ☁️  
    The notebook is optimized for **Google Colab**, eliminating the need for local installation.

---

## 🚀 How to Use

### 1️⃣ Open the Notebook in Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-zimmerle/whisper-transcription/blob/main/transcription_colab.ipynb)

### 2️⃣ Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

This allows access to video files and saves transcriptions directly to Google Drive.

### 3️⃣ Convert Videos to Audio

The notebook extracts audio from **`.mp4` files** and saves them in the configured folder. It also supports **`.mov` and `.avi` formats** (adjust the filter as needed).

```python
import os
import subprocess

input_folder = "/content/drive/My Drive/videos"
output_folder = "/content/drive/My Drive/audios"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.lower().endswith((".mp4", ".mov", ".avi")):  # Supports different video formats
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, os.path.splitext(file)[0] + ".m4a")
        
        cmd = ["ffmpeg", "-i", input_path, "-vn", "-acodec", "aac", "-b:a", "192k", output_path, "-y"]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("Conversion completed! Audio files saved in:", output_folder)
```

### 4️⃣ Check GPU Availability

Before running the transcription, it's recommended to enable a **GPU** in Google Colab for faster processing.

#### 🔹 How to Enable GPU in Google Colab: 
1. Click on **"Runtime"** in the menu. 
2. Select **"Change runtime type"**. 
3. Under **"Hardware accelerator"**, choose **"GPU"**. 
4. Click **"Save"**. 

Now, check if the GPU is available in your session:

```python
import torch

gpu_available = torch.cuda.is_available()
gpu_name = torch.cuda.get_device_name(0) if gpu_available else "No GPU detected"

print(f"GPU available: {gpu_available}")
print(f"GPU Name: {gpu_name}")
```

### 5️⃣ Install Whisper

```python
!pip install -q git+https://github.com/openai/whisper.git
!sudo apt update && sudo apt install ffmpeg
```

### 6️⃣ Load the Model and Transcribe Audio

The **'turbo' model** was chosen due to its excellent performance in both speed and accuracy.

```python
import os
import whisper

input_folder = "/content/drive/My Drive/audios"
output_folder = "/content/drive/My Drive/transcripts"

os.makedirs(output_folder, exist_ok=True)

model = whisper.load_model("turbo", device="cuda")

for file_name in os.listdir(input_folder):
    if file_name.endswith(".m4a"):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name.replace(".m4a", ".txt"))

        print(f"Transcribing: {file_name} using GPU...")
        
        # The language parameter ensures accurate transcription
        result = model.transcribe(input_path, language="pt")  # In this project, the default language is Portuguese (can be changed)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result["text"])

        print(f"Transcription saved at: {output_path}")

print("✅ Transcription completed! All files have been processed.")
```

---

## 📜 Dependencies

If you want to run the project locally, install the required dependencies:

```bash
pip install -r requirements.txt
```

**Content of `requirements.txt`:**

```
openai-whisper
torch
numpy
ffmpeg-python
```

---

## 🏆 Credits & Acknowledgments  

- This project utilizes **[Whisper](https://github.com/openai/whisper)**, an automatic speech recognition (ASR) model developed by OpenAI.  
- The notebook is designed to run on **[Google Colab](https://colab.research.google.com/)**, a cloud-based platform by Google Research that provides free GPU access for machine learning tasks.  
- **[FFmpeg](https://ffmpeg.org/)** is used for audio extraction and processing.  

---

📌 _This repository makes audio transcription easy using Whisper on Google Colab. Feel free to clone, modify, and contribute!_ 🚀