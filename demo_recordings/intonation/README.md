# Demo Recordings - Intonation

This directory contains pre-recorded utterances for the "Is This a Question?" pitch contour demo.

## File Naming

Name your files descriptively:
- `statement.wav` - Falling pitch (declarative)
- `question.wav` - Rising pitch (interrogative)
- `surprise.wav` - High rising terminal
- `flat.wav` - Monotone/level pitch

## Recommended Files

Create recordings that demonstrate different intonation patterns:

**Basic patterns:**
- `statement.wav` - "You're going to the store" (falling ⬇️)
- `question.wav` - "You're going to the store?" (rising ⬆️)
- `list.wav` - "Apples, oranges, and bananas" (fall-rise-fall)

**Advanced patterns:**
- `wh_question.wav` - "Where are you going?" (falling)
- `yes_no_question.wav` - "Are you going?" (rising)
- `emphasis.wav` - "I said NO" (sharp rise-fall)
- `uncertainty.wav` - "Maybe...?" (slight rise)

## Recording Tips

1. **Use the same sentence** for statement vs. question comparisons
2. **Exaggerate slightly** to make patterns clear for demonstration
3. **Speak naturally** but clearly
4. **Keep volume consistent** across recordings
5. **Record in a quiet space** to minimize noise
6. **Save as WAV format** (16-bit, 44.1kHz recommended)

## Usage in App

In the Streamlit app:
1. Navigate to "❓ Is This a Question?" demo
2. Click "📁 Load Demo Recordings" expander
3. Click **📥** next to files to load them individually
4. All loaded recordings will be plotted together for comparison
5. Maximum 3 recordings can be displayed at once

## Perfect for Demonstrating

- **Statement vs. Question** - Same words, different meaning
- **Cross-linguistic differences** - How different languages use intonation
- **Pragmatic effects** - Sarcasm, surprise, uncertainty
- **Regional variations** - Different dialect intonation patterns

## Example Workflow

```bash
# Record your utterances using Audacity, Praat, or the app itself
# Save them in this directory:
# - statement.wav
# - question.wav
# - surprise.wav

# In the app, load all three and see them compared side-by-side!
```

## Technical Requirements

- **Format:** WAV files (.wav or .WAV)
- **Duration:** 2-6 seconds recommended
- **Content:** Clear speech with distinct intonation patterns
- **Limit:** Load up to 3 files for comparison
