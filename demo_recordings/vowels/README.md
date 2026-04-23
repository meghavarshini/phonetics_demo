# Demo Recordings - Vowels

This directory contains pre-recorded vowel samples that can be loaded into the Vowel Plotting demo.

## Current Files

This directory contains the following demo recordings:

**Front vowels:**
- `i.wav` (heed)
- `ɪ.wav` (hid)
- `e.wav` (hayed)
- `ɛ.wav` (head)
- `æ.wav` (had)

**Back vowels:**
- `u.wav` (who'd)
- `ʊ.wav` (hood)
- `o.wav` (hoed)
- `ɔ.wav` (hawed)
- `ɑ.wav` (hod)

## File Naming Convention

Files can be named in two ways:
- **Simple format:** `<label>.wav` - e.g., `i.wav`, `æ.wav` (currently used)
- **Prefix format:** `vowel_<label>.wav` - e.g., `vowel_i.wav`, `vowel_æ.wav`

The app will automatically extract the label from the filename and display it on the vowel chart.

## Adding More Files

To add additional vowel recordings:

**Central vowels (not currently included):**
- `ə.wav` (about)
- `ʌ.wav` (hut)
- `ɜ.wav` (bird)

1. **Record in a quiet environment** to avoid background noise
2. **Hold each vowel steady** for 1-2 seconds
3. **Use a consistent volume** across all recordings
4. **Save as WAV format** (16-bit, 44.1kHz recommended)
5. **Say only the vowel**, not the entire word
6. **Name the file** with the IPA symbol (e.g., `ə.wav`)

## Using Demo Files

In the Streamlit app:
1. Go to the "🗣️ Vowel Plotting" demo
2. Click "📁 Load Demo Recordings" expander
3. Click **📥** next to individual files to load them, or
4. Click "📥 Load All Demo Files" to load all at once

Demo files are perfect for:
- Quick demonstrations without live recording
- Backup files when microphone fails
- Showing reference vowel spaces
- Comparing different speakers or dialects

## Technical Details

- **Format:** WAV files (.wav or .WAV)
- **Processing:** The app automatically extracts F1 and F2 formants
- **Integration:** Loaded files are treated the same as live recordings
- **Mixing:** You can combine demo files and live recordings on the same chart

## Example Workflow

```bash
# These demo files are already loaded and ready to use!
# In the app:
# 1. Open "📁 Load Demo Recordings"
# 2. Click "📥 Load All Demo Files"
# 3. See all 10 vowels plotted instantly!
```
