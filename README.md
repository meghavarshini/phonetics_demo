# 🎙️ Phonetics Demo - Maryland Day

Interactive phonetics demonstrations for exploring speech sounds through real-time audio analysis.

## Features

- **📊 Live Spectrogram & Waveform**: Visualize your voice as waveforms and spectrograms
- **❓ Is This a Question?**: Compare pitch contours to see how intonation conveys meaning
- **🗣️ Vowel Plotting**: Map your vowel space on an F1-F2 chart

## Live Demo

🌐 **[Try it live!](#)** *(Add your Streamlit Cloud URL here after deployment)*

## Running Locally

### Prerequisites

- Python 3.9, 3.10, or 3.11
- Microphone access

### Installation

```bash
# Clone the repository
git clone https://github.com/mkswamy/phonetics_demo.git
cd phonetics_demo

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run phonetics_app_multi_vowel.py
```

## How It Works

### Vowel Analysis
Uses **Praat's formant extraction** to identify F1 and F2 frequencies, which correspond to tongue height and frontness/backness.

### Pitch Tracking
Employs **autocorrelation** with voicing detection and energy thresholding to extract pitch contours while filtering out noise.

### Visualizations
- **Spectrograms**: Generated with librosa's STFT
- **Waveforms**: Raw amplitude visualization
- **Vowel Charts**: F1-F2 plotting with standard reference vowels

## Technologies

- **Streamlit**: Web interface
- **Parselmouth**: Praat integration for phonetic analysis
- **Librosa**: Audio processing and visualization
- **NumPy/SciPy**: Signal processing
- **Matplotlib**: Chart generation

## Maryland Day

This demo was created for **Maryland Day** to introduce visitors to the science of phonetics and speech analysis.

## License

MIT License - See LICENSE file for details

## Credits

Developed for the University of Maryland's Maryland Day celebration.
