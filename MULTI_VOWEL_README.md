# Phonetics Demo App - Multi-Vowel Version

## What's New

The updated `phonetics_app_multi_vowel.py` now supports **recording multiple vowels** with improved features:

### Key Features

1. **Multi-Vowel Recording**
   - Record as many vowels as you want
   - Each recording is labeled and saved as a separate WAV file in `recordings/` folder
   - All vowels are plotted together on one chart

2. **IPA Symbol Helper**
   - Expandable panel with common IPA vowel symbols
   - **Front vowels:** `i` `ɪ` `e` `ɛ` `æ`
   - **Central vowels:** `ə` `ʌ` `ɜ`
   - **Back vowels:** `u` `ʊ` `o` `ɔ` `ɑ`
   - Copy-paste directly into the label field!

3. **Recording Management**
   - View list of all recorded vowels
   - Delete individual vowels
   - Clear all recordings at once
   - See F1/F2 values for each recording

4. **Auto-Save**
   - Files saved as: `recordings/vowel_{label}_{timestamp}.wav`
   - Example: `recordings/vowel_i_20260423_143052.wav`

5. **Enhanced Visualization**
   - Multiple vowels plotted with different colors
   - Automatic label staggering to avoid overlap
   - Download button for the final chart (300 DPI PNG)

## Required Libraries

### Python Version
- **Python 3.9, 3.10, or 3.11** (all versions supported)

### Core Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| **streamlit** | Latest | Web app framework for interactive demos |
| **praat-parselmouth** | Latest | Acoustic analysis (formants, pitch extraction) |
| **librosa** | Latest | Audio processing and spectrogram generation |
| **sounddevice** | Latest | Real-time audio recording from microphone |
| **soundfile** | Latest | Reading/writing WAV audio files |
| **numpy** | Latest | Numerical computation and array operations |
| **scipy** | Latest | Scientific computing (used by librosa) |
| **matplotlib** | Latest | Plotting and visualization |

### Installation

**Option 1: Using requirements.txt (Easiest)**
```bash
# Create virtual environment
python3.9 -m venv phonetics_env
source phonetics_env/bin/activate  # On Windows: phonetics_env\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

**Option 2: Using Conda (Recommended for Mac)**
```bash
# Create environment
conda create -n phonetics python=3.9 -y
conda activate phonetics

# Install from conda-forge
conda install -c conda-forge numpy scipy matplotlib pandas -y

# Install remaining with pip
pip install streamlit praat-parselmouth librosa sounddevice soundfile
```

**Option 3: Using pip only**
```bash
# Create virtual environment
python3.9 -m venv phonetics_env
source phonetics_env/bin/activate  # On Windows: phonetics_env\Scripts\activate

# Install all packages
pip install streamlit praat-parselmouth librosa sounddevice soundfile numpy scipy matplotlib
```

**Option 4: Google Colab**
```python
# Run in a Colab cell
!pip install streamlit praat-parselmouth librosa sounddevice soundfile
```

### Verify Installation
```bash
python -c "import streamlit, parselmouth, librosa, sounddevice; print('✅ All packages installed')"
```

### Troubleshooting

**If microphone doesn't work:**
```bash
# Test audio devices
python -m sounddevice
```

**If praat-parselmouth fails to install:**
- Make sure you have a C++ compiler installed
- On Mac: `xcode-select --install`
- On Windows: Install Visual Studio Build Tools

**If librosa is slow to install:**
- It has many dependencies; be patient or use conda which has pre-built binaries

## Usage

```bash
# Run the multi-vowel version
streamlit run phonetics_app_multi_vowel.py
```

## Workflow

1. **Select vowel symbol** from the IPA helper (or type your own)
2. **Record** your vowel
3. **Review** F1/F2 values and audio playback
4. **Repeat** for as many vowels as you want
5. **View** all vowels plotted together on the chart
6. **Download** your personalized vowel chart

## Comparison with Original

| Feature | Original (`phonetics_app.py`) | Multi-Vowel (`phonetics_app_multi_vowel.py`) |
|---------|-------------------------------|----------------------------------------------|
| Vowels per session | 1 (overwrites) | Unlimited |
| Save audio files | ❌ No | ✅ Yes (auto-saves to `recordings/`) |
| IPA symbols | ❌ No helper | ✅ Built-in copy-paste panel |
| Vowel management | N/A | ✅ List, delete, clear all |
| Chart download | ❌ No | ✅ PNG export |
| Colors | Single | 10 distinct colors |

## For Maryland Day

The multi-vowel version is **perfect for public demos** where you want to:
- Compare multiple participants' vowel spaces
- Show dialect differences (e.g., different pronunciations of "cot" vs "caught")
- Build up a complete vowel inventory live
- Keep recordings for later analysis

## Files Structure

```
maryland_day_demo/
├── phonetics_demos.ipynb          # Notebook with all code
├── phonetics_app.py               # Original single-vowel version
├── phonetics_app_multi_vowel.py   # NEW: Multi-vowel version ⭐
└── recordings/                     # Auto-created directory for WAV files
    ├── vowel_i_20260423_143052.wav
    ├── vowel_æ_20260423_143105.wav
    └── ...
```

## Tips for Presenters

- **Pre-record a few vowels** before the demo starts to show the chart immediately
- **Have participants say the same vowel** (e.g., everyone says "ee") to show individual variation
- **Compare languages** - have multilingual speakers record the same vowel
- **Show vowel shifts** - have the same person say a vowel in different contexts

Enjoy! 🎉
