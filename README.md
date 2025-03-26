# 🎙️ Whisperizer-Colab

**Whisperizer-Colab** is a modular, Google Colab-friendly notebook for automatically transcribing audio extracted from video files using OpenAI’s **Whisper** model.  
It combines the power of **FFmpeg** for audio processing with **Whisper** for high-quality speech recognition — optimized for GPU usage when available.

---

## ✨ Features

- **Extract Audio from Video Files**  
  Uses **FFmpeg** (CPU-based) to extract audio from `.mp4`, `.mov`, `.avi`, and `.mkv` files.

- **Automatic Transcription with Whisper**  
  Leverages **OpenAI's Whisper** (with GPU support) for accurate and efficient transcription. The `"turbo"` model is loaded by default.

- **Optimized for Google Colab**  
  Step-by-step workflow includes mounting Google Drive and checking for GPU availability to speed up transcription.

---

## 🚀 How to Use

1. **Open the Notebook in Google Colab**  
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-zimmerle/Whisperizer-Colab/blob/main/transcription_colab.ipynb)

2. **Notebook Structure**  
   The notebook is organized into **six main cells**:

   1. **Mount Google Drive**  
      ```python
      from google.colab import drive
      drive.mount('/content/drive')
      ```
      This allows you to read and write files directly to Google Drive.

   2. **Install Dependencies**  
      ```python
      !sudo apt-get update -qq
      !sudo apt-get install -y ffmpeg
      !pip install -q git+https://github.com/openai/whisper.git
      ```
      Installs FFmpeg and the Whisper library.

   3. **Imports and GPU Check**  
      ```python
      import os
      import subprocess
      import torch
      import whisper

      gpu_available = torch.cuda.is_available()
      gpu_name = torch.cuda.get_device_name(0) if gpu_available else "No GPU detected"
      device = "cuda" if gpu_available else "cpu"

      print(f"GPU available for Whisper: {gpu_available}")
      print(f"GPU name: {gpu_name}")
      print(f"Whisper device: {device}")
      ```
      This checks if a GPU is available for Whisper. FFmpeg remains on CPU for audio extraction.

   4. **Function: `convert_videos_to_audio`**  
      ```python
      def convert_videos_to_audio(input_folder, output_folder):
          """
          Convert video files in the supported formats to M4A audio using FFmpeg (CPU).
          """
          os.makedirs(output_folder, exist_ok=True)

          supported_formats = (".mp4", ".mov", ".avi", ".mkv")

          for file in os.listdir(input_folder):
              if file.lower().endswith(supported_formats):
                  input_path = os.path.join(input_folder, file)
                  output_name = os.path.splitext(file)[0] + ".m4a"
                  output_path = os.path.join(output_folder, output_name)

                  print(f"Extracting audio from: {file}")

                  cmd = [
                      "ffmpeg",
                      "-i", input_path,
                      "-vn",
                      "-acodec", "aac",
                      "-b:a", "192k",
                      output_path,
                      "-y"
                  ]
                  subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

          print("\n✅ Audio extraction completed! Check your output folder.")
      ```

   5. **Function: `transcribe_audios`**  
      ```python
      def transcribe_audios(input_folder, output_folder, whisper_model="turbo", language="pt"):
          """
          Transcribe all .m4a audio files in 'input_folder' using Whisper (GPU if available),
          then save the transcripts as .txt files in 'output_folder'.
          """
          os.makedirs(output_folder, exist_ok=True)

          print(f"\nLoading Whisper model: {whisper_model}")
          model = whisper.load_model(whisper_model, device=device)
          print(f"✅ Whisper '{whisper_model}' model loaded successfully!\n")

          for file_name in os.listdir(input_folder):
              if file_name.lower().endswith(".m4a"):
                  input_path = os.path.join(input_folder, file_name)
                  output_text_path = os.path.join(output_folder, file_name.replace(".m4a", ".txt"))

                  print(f"Transcribing: {file_name} ...")

                  result = model.transcribe(input_path, language=language)

                  with open(output_text_path, "w", encoding="utf-8") as f:
                      f.write(result["text"])

                  print(f"Transcription saved to: {output_text_path}")

          print("\n✅ All transcriptions completed successfully!")
      ```

---

## 📄 Dependencies

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

Feel free to clone, modify, and contribute to this project. Your feedback is always welcome!

Happy transcribing! 🚀
