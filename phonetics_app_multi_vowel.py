import streamlit as st
import numpy as np
from scipy.signal import medfilt
import parselmouth
from parselmouth.praat import call
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import io
import os
import time
from datetime import datetime

# Suppress warnings and verbose output
import warnings
warnings.filterwarnings('ignore')

# Suppress numpy warnings
np.seterr(all='ignore')

# Suppress matplotlib warnings
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# Configure page
st.set_page_config(
    page_title="Phonetics Demo - Maryland Day",
    page_icon="🎙️",
    layout="wide"
)

# Color scheme - coordinated colors for visualizations
COLORS = {
    'primary': '#FF6B6B',      # Coral red for waveforms
    'secondary': '#4ECDC4',    # Teal for spectrograms
    'accent': '#FFE66D',       # Yellow for highlights
    'dark': '#2C3E50',         # Dark blue-gray
    'light': '#ECF0F1'         # Light gray
}

# Color palette for multiple vowels
VOWEL_COLORS = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181', '#AA96DA',
                '#FCBAD3', '#A8D8EA', '#FFABAB', '#FFC8A2', '#D4A5A5']

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def record_audio(duration=3, sample_rate=44100):
    """Record audio from the microphone with visual countdown timer.

    Args:
        duration (int): Recording duration in seconds
        sample_rate (int): Sampling rate in Hz

    Returns:
        tuple: (audio_data, sample_rate)
    """
    # Start recording in a non-blocking way
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float64'
    )

    # Show countdown timer
    status_placeholder = st.empty()
    progress_bar = st.progress(0)

    status_placeholder.success("🔴 Recording started!")

    # Update countdown every 0.1 seconds
    steps = int(duration * 10)  # 10 updates per second
    for i in range(steps + 1):
        elapsed = i / 10.0
        remaining = duration - elapsed
        progress = elapsed / duration

        # Update progress bar
        progress_bar.progress(min(progress, 1.0))

        # Update status message
        if remaining > 0.1:
            status_placeholder.info(f"🎤 Recording... {remaining:.1f}s remaining")
        else:
            status_placeholder.success("✅ Recording complete!")

        time.sleep(0.1)

    # Wait for recording to finish
    sd.wait()

    # Clear the progress bar and status
    progress_bar.empty()
    status_placeholder.empty()

    return audio.flatten(), sample_rate


def save_audio_file(audio, sample_rate, label):
    """Save audio to a WAV file.

    Args:
        audio (np.array): Audio signal
        sample_rate (int): Sampling rate
        label (str): Vowel label for filename

    Returns:
        str: Filename of saved audio
    """
    # Create recordings directory if it doesn't exist
    os.makedirs('recordings', exist_ok=True)

    # Generate filename with timestamp and label
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = label.replace(' ', '_').replace('/', '_')
    filename = f"recordings/vowel_{safe_label}_{timestamp}.wav"

    # Save as WAV file
    sf.write(filename, audio, sample_rate)

    return filename


def load_audio_file(filepath):
    """Load audio from a WAV file.

    Args:
        filepath (str): Path to the WAV file

    Returns:
        tuple: (audio_data, sample_rate) or (None, None) if error
    """
    try:
        audio, sr = sf.read(filepath)
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        return audio, sr
    except Exception as e:
        st.error(f"Error loading {filepath}: {e}")
        return None, None


def get_demo_recordings():
    """Get list of demo recording files from ./demo_recordings/vowels/

    Returns:
        list: List of tuples (filename, full_path)
    """
    demo_dir = "./demo_recordings/vowels"
    if not os.path.exists(demo_dir):
        return []

    wav_files = []
    for filename in os.listdir(demo_dir):
        if filename.endswith('.wav') or filename.endswith('.WAV'):
            full_path = os.path.join(demo_dir, filename)
            wav_files.append((filename, full_path))

    return sorted(wav_files)


def extract_formants(audio, sample_rate, num_formants=5):
    """Extract formant frequencies using Parselmouth/Praat.

    Args:
        audio (np.array): Audio signal
        sample_rate (int): Sampling rate
        num_formants (int): Number of formants to extract

    Returns:
        dict: Dictionary with F1, F2, F3, etc.
    """
    # Create Praat Sound object
    sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)

    # Extract formants using Praat
    formant = call(sound, "To Formant (burg)", 0.0, num_formants, 5500, 0.025, 50)

    # Get formant values at the middle of the sound
    duration = sound.duration
    mid_point = duration / 2.0

    formants = {}
    for i in range(1, num_formants + 1):
        try:
            f_val = call(formant, "Get value at time", i, mid_point, "Hertz", "Linear")
            if not np.isnan(f_val):
                formants[f"F{i}"] = f_val
        except:
            pass

    return formants


def extract_pitch(audio, sample_rate):
    """Extract pitch contour using Parselmouth/Praat with voicing detection.

    Args:
        audio (np.array): Audio signal
        sample_rate (int): Sampling rate

    Returns:
        tuple: (time_points, pitch_values)
    """
    sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)

    # Extract pitch using autocorrelation with stricter voicing threshold
    # Parameters: time_step, pitch_floor, pitch_ceiling
    # Using default "To Pitch" but will filter by voicing strength
    pitch = call(sound, "To Pitch", 0.0, 75, 600)  # 75-600 Hz range for human speech

    # Calculate energy threshold (20% of max RMS energy) - more permissive
    # Compute RMS energy in windows
    window_size = int(0.03 * sample_rate)  # 30ms windows
    hop_size = int(0.01 * sample_rate)  # 10ms hops
    energies = []
    energy_times = []

    for i in range(0, len(audio) - window_size, hop_size):
        window = audio[i:i + window_size]
        rms = np.sqrt(np.mean(window ** 2))
        energies.append(rms)
        energy_times.append(i / sample_rate + (window_size / 2 / sample_rate))

    energies = np.array(energies)
    energy_times = np.array(energy_times)

    # More permissive thresholds
    energy_threshold = np.max(energies) * 0.15 if len(energies) > 0 else 0  # 15% of maximum
    voicing_threshold = 0.4  # 40% voicing confidence (more permissive)

    # Get pitch values over time with voicing and energy filtering
    time_points = []
    pitch_values = []

    for t in np.arange(0, sound.duration, 0.01):  # Sample every 10ms
        pitch_value = call(pitch, "Get value at time", t, "Hertz", "Linear")

        # Check if pitch exists
        if np.isnan(pitch_value) or pitch_value <= 0:
            continue

        # Check voicing strength (Praat's internal voicing decision)
        try:
            voicing_strength = call(pitch, "Get strength", t)
            if voicing_strength < voicing_threshold:
                continue
        except:
            # If voicing strength can't be computed, use energy check only
            pass

        # Check energy threshold
        # Find closest energy measurement
        if len(energy_times) > 0:
            closest_idx = np.argmin(np.abs(energy_times - t))
            if energies[closest_idx] < energy_threshold:
                continue

        time_points.append(t)
        pitch_values.append(pitch_value)

    # Post-processing: remove isolated spikes using median filter
    # Only apply if we have enough points
    if len(pitch_values) > 5:
        pitch_values = np.array(pitch_values)
        time_points = np.array(time_points)

        # Apply median filter to remove spikes (window size 5)
        pitch_values_filtered = medfilt(pitch_values, kernel_size=5)

        # Remove points that jump more than 50% from the filtered value (octave errors)
        # More permissive threshold
        valid_indices = []
        for i in range(len(pitch_values)):
            if i == 0:
                valid_indices.append(i)
            else:
                ratio = abs(pitch_values[i] - pitch_values_filtered[i]) / pitch_values_filtered[i]
                if ratio < 0.4:  # Within 40% of median-filtered value (more permissive)
                    valid_indices.append(i)

        if len(valid_indices) > 0:
            time_points = time_points[valid_indices]
            pitch_values = pitch_values_filtered[valid_indices]
        # If filtering removed everything, return unfiltered data
        else:
            pitch_values = pitch_values_filtered

    return np.array(time_points), np.array(pitch_values)


def plot_vowel_chart_multi(vowel_recordings=None):
    """Plot multiple vowels on a standard F1-F2 vowel chart.

    Args:
        vowel_recordings (list): List of dicts with 'f1', 'f2', 'label' keys

    Returns:
        matplotlib.figure.Figure: The vowel chart figure
    """
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')

    # Reference vowels (approximate American English values)
    reference_vowels = {
        'i (heed)': (280, 2250),
        'ɪ (hid)': (400, 1900),
        'e (hayed)': (400, 2100),
        'ɛ (head)': (550, 1800),
        'æ (had)': (700, 1700),
        'ɑ (hard)': (700, 1100),
        'ɔ (hawed)': (600, 900),
        'o (hoed)': (450, 850),
        'ʊ (hood)': (400, 1000),
        'u (who\'d)': (300, 900)
    }

    # Plot reference vowels (lighter, in background)
    for vowel, (f1_ref, f2_ref) in reference_vowels.items():
        ax.scatter(f2_ref, f1_ref, s=200, alpha=0.2, c=COLORS['secondary'],
                  edgecolors=COLORS['dark'], linewidth=1.5)
        ax.annotate(vowel, (f2_ref, f1_ref), fontsize=14, ha='center',
                   va='center', weight='bold', alpha=0.5)

    # Plot user's vowels
    if vowel_recordings:
        for idx, vowel in enumerate(vowel_recordings):
            color = VOWEL_COLORS[idx % len(VOWEL_COLORS)]
            f1 = vowel['f1']
            f2 = vowel['f2']
            label = vowel['label']

            # Plot the vowel point
            ax.scatter(f2, f1, s=400, c=color, marker='*',
                      edgecolors=COLORS['dark'], linewidth=2.5, zorder=10,
                      label=label)

            # Add label with offset to avoid overlap
            offset_y = 25 + (idx % 3) * 15  # Stagger labels
            ax.annotate(label, (f2, f1), xytext=(0, offset_y),
                       textcoords='offset points', fontsize=14,
                       weight='bold', color=color,
                       ha='center',
                       bbox=dict(boxstyle='round,pad=0.5',
                                facecolor='white',
                                edgecolor=color,
                                linewidth=2,
                                alpha=0.9))

    # Invert axes (phonetic convention)
    ax.invert_xaxis()
    ax.invert_yaxis()

    # Labels and formatting
    ax.set_xlabel('F2 (Hz)', fontsize=14, weight='bold')
    ax.set_ylabel('F1 (Hz)', fontsize=14, weight='bold')

    title = 'Vowel Space Chart (F1 vs F2)'
    if vowel_recordings:
        title += f' - {len(vowel_recordings)} vowel(s) recorded'
    ax.set_title(title, fontsize=16, weight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--')

    if vowel_recordings:
        ax.legend(fontsize=13, loc='lower right', framealpha=0.9)

    plt.tight_layout()
    return fig


def plot_pitch_contour(time, pitch):
    """Plot pitch contour over time.

    Args:
        time (np.array): Time points in seconds
        pitch (np.array): Pitch values in Hz

    Returns:
        matplotlib.figure.Figure: The pitch contour figure
    """
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')

    # Plot pitch contour
    ax.plot(time, pitch, linewidth=3, color=COLORS['primary'], label='Pitch')
    ax.fill_between(time, pitch, alpha=0.3, color=COLORS['primary'])

    # Determine if rising or falling
    if len(pitch) > 2:
        # Compare first third vs last third
        third = len(pitch) // 3
        start_avg = np.mean(pitch[:third])
        end_avg = np.mean(pitch[-third:])

        if end_avg > start_avg * 1.1:  # 10% threshold
            pattern = "Rising (Question-like) ⬆️"
            color = COLORS['secondary']
        elif end_avg < start_avg * 0.9:
            pattern = "Falling (Statement-like) ⬇️"
            color = COLORS['accent']
        else:
            pattern = "Level (Flat) ➡️"
            color = COLORS['dark']

        ax.set_title(f'Pitch Contour: {pattern}', fontsize=16, weight='bold',
                    pad=20, color=color)
    else:
        ax.set_title('Pitch Contour', fontsize=16, weight='bold', pad=20)

    # Labels and formatting
    ax.set_xlabel('Time (s)', fontsize=14, weight='bold')
    ax.set_ylabel('Pitch (Hz)', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=13)

    plt.tight_layout()
    return fig


def plot_pitch_contours_multi(pitch_recordings=None):
    """Plot multiple pitch contours for comparison.

    Args:
        pitch_recordings (list): List of dicts with 'time', 'pitch', 'label' keys

    Returns:
        matplotlib.figure.Figure: The pitch contour comparison figure
    """
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')

    # Define colors for pitch contours (up to 3) - colorblind accessible
    pitch_colors = ['blue', 'red', 'green']

    # Define markers for colorblind accessibility
    pitch_markers = ['s', '^', 'o']  # square, triangle, circle
    marker_labels = ['■', '▲', '●']  # Unicode symbols for legend

    if pitch_recordings:
        # Plot each recording with different color and marker
        for idx, recording in enumerate(pitch_recordings):
            color = pitch_colors[idx % len(pitch_colors)]
            marker = pitch_markers[idx % len(pitch_markers)]
            marker_label = marker_labels[idx % len(marker_labels)]
            time = recording['time']
            pitch = recording['pitch']
            label = recording['label']

            # Plot pitch contour with markers for colorblind accessibility
            ax.plot(time, pitch, linewidth=5, color=color,
                   marker=marker, markersize=8, markevery=len(time)//20 or 1,
                   label=f"{marker_label} {label}")

        # Labels and formatting
        ax.set_xlabel('Time (s)', fontsize=14, weight='bold')
        ax.set_ylabel('Pitch (Hz)', fontsize=14, weight='bold')
        ax.set_title(f'Pitch Contour Comparison - {len(pitch_recordings)} recording(s)',
                    fontsize=16, weight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=12, loc='best', framealpha=0.9)
    else:
        # Empty plot with instructions
        ax.text(0.5, 0.5, 'No recordings yet\nRecord up to 3 utterances to compare',
                ha='center', va='center', fontsize=14, color='gray',
                transform=ax.transAxes)
        ax.set_xlabel('Time (s)', fontsize=14, weight='bold')
        ax.set_ylabel('Pitch (Hz)', fontsize=14, weight='bold')
        ax.set_title('Pitch Contour Comparison', fontsize=16, weight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def plot_spectrograms_with_pitch(pitch_recordings):
    """Plot spectrograms with overlaid pitch contours for each recording.

    Args:
        pitch_recordings (list): List of dicts with 'audio', 'sample_rate', 'time', 'pitch', 'label' keys

    Returns:
        matplotlib.figure.Figure: Figure with stacked spectrograms
    """
    if not pitch_recordings:
        return None

    # Define colors for pitch contours (matches the pitch plot) - colorblind accessible
    pitch_colors = ['blue', 'red', 'green']

    # Define markers for colorblind accessibility
    pitch_markers = ['s', '^', 'o']  # square, triangle, circle
    marker_labels = ['■', '▲', '●']  # Unicode symbols for legend

    # Create subplots - one per recording
    n_recordings = len(pitch_recordings)
    fig, axes = plt.subplots(n_recordings, 1, figsize=(14, 5 * n_recordings), facecolor='white')

    # Handle single recording case (axes is not a list)
    if n_recordings == 1:
        axes = [axes]

    for idx, (recording, ax) in enumerate(zip(pitch_recordings, axes)):
        audio = recording['audio']
        sr = recording['sample_rate']
        time_points = recording['time']
        pitch_values = recording['pitch']
        label = recording['label']
        color = pitch_colors[idx % len(pitch_colors)]
        marker = pitch_markers[idx % len(pitch_markers)]
        marker_label = marker_labels[idx % len(marker_labels)]

        # Compute spectrogram using librosa
        D = librosa.stft(audio)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # Plot spectrogram
        img = librosa.display.specshow(
            S_db,
            sr=sr,
            x_axis='time',
            y_axis='hz',
            ax=ax,
            cmap='gray_r'  # White to black (better for overlay)
        )

        # Overlay pitch contour on the spectrogram with markers
        # Create a second y-axis for pitch
        ax2 = ax.twinx()
        ax2.plot(time_points, pitch_values, linewidth=4, color=color,
                marker=marker, markersize=8, markevery=len(time_points)//20 or 1,
                label=f'{marker_label} Pitch: {label}')
        ax2.set_ylabel('Pitch (Hz)', fontsize=12, weight='bold', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim([50, 500])  # Reasonable pitch range

        # Format the main axis
        ax.set_xlabel('Time (s)', fontsize=12, weight='bold')
        ax.set_ylabel('Frequency (Hz)', fontsize=12, weight='bold')
        ax.set_title(f'{marker_label} {label}', fontsize=14, weight='bold', pad=10, color=color)
        ax.set_ylim([0, 4000])  # Focus on speech range

        # Add legend
        ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    return fig


def plot_waveform_and_spectrogram(audio, sample_rate):
    """Plot waveform and spectrogram with coordinated colors.

    Args:
        audio (np.array): Audio signal
        sample_rate (int): Sampling rate

    Returns:
        matplotlib.figure.Figure: Combined waveform and spectrogram figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), facecolor='white')

    # Time array
    time = np.linspace(0, len(audio) / sample_rate, len(audio))

    # === Waveform ===
    ax1.plot(time, audio, linewidth=1, color=COLORS['primary'], alpha=0.8)
    ax1.fill_between(time, audio, alpha=0.3, color=COLORS['primary'])
    ax1.set_xlabel('Time (s)', fontsize=12, weight='bold')
    ax1.set_ylabel('Amplitude', fontsize=12, weight='bold')
    ax1.set_title('Waveform', fontsize=14, weight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim([0, time[-1]])

    # === Spectrogram ===
    # Compute spectrogram using librosa
    D = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Plot spectrogram with custom colormap
    img = librosa.display.specshow(
        S_db,
        sr=sample_rate,
        x_axis='time',
        y_axis='hz',
        ax=ax2,
        cmap='viridis'  # Beautiful teal-to-yellow colormap
    )

    ax2.set_xlabel('Time (s)', fontsize=12, weight='bold')
    ax2.set_ylabel('Frequency (Hz)', fontsize=12, weight='bold')
    ax2.set_title('Spectrogram', fontsize=14, weight='bold', pad=15)
    ax2.set_ylim([0, 8000])  # Focus on speech range

    # Add colorbar
    cbar = fig.colorbar(img, ax=ax2, format='%+2.0f dB')
    cbar.set_label('Intensity (dB)', fontsize=11, weight='bold')

    plt.tight_layout()
    return fig


# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    """Main Streamlit app with three demo modes."""

    # Initialize session state for vowel recordings
    if 'vowel_recordings' not in st.session_state:
        st.session_state.vowel_recordings = []

    # Initialize session state for pitch recordings (Demo 2)
    if 'pitch_recordings' not in st.session_state:
        st.session_state.pitch_recordings = []

    # Header
    st.title("🎙️ Phonetics Demo - Maryland Day")
    st.markdown("""
    ### Explore the Science of Speech!
    Choose a demo below to analyze your voice in real-time.
    """)

    # Sidebar for demo selection
    st.sidebar.title("Demo Selection")
    demo_mode = st.sidebar.radio(
        "Choose a demo:",
        ["🏠 Home", "📊 Live Spectrogram & Waveform", "❓ Is This a Question?", "🗣️ Vowel Plotting"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Tips:**
    - Speak clearly into your microphone
    - Find a quiet environment
    - Try different sounds and see what happens!
    """)

    # =============================================================================
    # LANDING PAGE / HOME
    # =============================================================================
    if demo_mode == "🏠 Home":
        st.header("🎉 Welcome to Maryland Day!")
        st.markdown("""
        ### Explore the Science of Speech with Interactive Phonetics Demos

        Welcome to our interactive phonetics demonstration! Today, you'll get to explore
        the fascinating science behind how we produce and perceive speech sounds.
        """)

        st.markdown("---")

        # Introduction to Phonetics
        st.subheader("🔬 What is Phonetics?")
        st.markdown("""
        **Phonetics** is the scientific study of how we produce, perceive, and distinguish speech sounds. 
                    It explores:

        - **How we produce sounds** using our articulators (tongue, lips, vocal cords, etc.)
        - **The physical properties** of speech sounds (frequency, amplitude, duration)
        - **How we perceive and distinguish** different sounds (like "cat" vs "cot")

        Every time you speak, your body performs an incredibly complex choreography!
        Your vocal cords vibrate, your tongue moves to precise positions, and your mouth
        shapes the sound waves that travel to a listener's ears. Checkout how
        phoneticians analyse phonetic data!
        """)

        st.markdown("---")

        # External Resources
        st.subheader("🌐 Explore More: Interactive Phonetics Resources")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🗣️ [Pink Trombone](https://dood.al/pinktrombone/)")
            st.markdown("""
            **Interactive vocal tract simulator**

            Pink Trombone is a fantastic hands-on tool that lets you control a
            virtual vocal tract in real-time. You can:
            - Move a virtual tongue to different positions
            - Control voicing (vocal cord vibration)
            - Hear exactly how different tongue positions create different vowel sounds
            - Experiment with consonants by creating constrictions

            It's like having an X-ray view of speech production! Perfect for understanding
            how tiny changes in tongue position create completely different sounds.
            """)

        with col2:
            st.markdown("#### 📊 [Interactive IPA Chart](https://www.ipachart.com)")
            st.markdown("""
            **International Phonetic Alphabet reference**

            The IPA Chart is the linguist's best friend! This interactive resource:
            - Shows all the sounds used in human languages
            - Lets you click any symbol to hear how it sounds
            - Organizes sounds by how and where they're produced
            - Includes both common and rare speech sounds from languages worldwide

            The IPA is a standardized system for transcribing speech sounds. Whether you're
            learning a new language or studying linguistics, this chart is essential!
            """)

        st.markdown("---")

        # Demo Introductions
        st.subheader("🎯 Today's Interactive Demos")
        st.markdown("""
        Choose a demo from the sidebar to get started! Here's what each one offers:
        """)

        # Demo 1
        st.markdown("#### 📊 1. Live Spectrogram & Waveform")
        st.markdown("""
        **See your voice as visual patterns**

        This demo transforms your speech into two types of visualizations:
        - **Waveform**: Shows how the amplitude (loudness) of your voice changes over time
        - **Spectrogram**: Reveals the frequency content of your speech, like a musical score

        You'll discover that:
        - Vowels create clear harmonic patterns (horizontal stripes)
        - Consonants like [s] and [sh] create noise patterns at different frequencies
        - Singing shows up as perfectly straight horizontal lines
        - Whispering has no harmonics, just noise!

        *Great for: Understanding the difference between voiced and voiceless sounds*
        """)

        # Demo 2
        st.markdown("#### ❓ 2. Is This a Question?")
        st.markdown("""
        **Compare pitch patterns in speech**

        How do we know when someone is asking a question versus making a statement?
        Often, it's all about the pitch contour (how pitch rises or falls)!

        This demo lets you:
        - Record up to 3 different utterances
        - Visualize their pitch contours side-by-side
        - See spectrograms with pitch overlaid
        - Compare statements (falling pitch) vs. questions (rising pitch)

        Try saying "You're going to the store" as both a statement and a question
        to see the dramatic difference in pitch patterns!

        *Great for: Understanding intonation and how we use pitch to convey meaning*
        """)

        # Demo 3
        st.markdown("#### 🗣️ 3. Vowel Plotting")
        st.markdown("""
        **Map your vowel space**

        Vowels are primarily distinguished by their **formants** – resonant frequencies
        created by the shape of your vocal tract. This demo:

        - Records your vowel sounds and extracts their first two formants (F1 and F2)
        - Plots them on a vowel chart (F1 vs F2)
        - Compares your vowels to standard reference vowels
        - Lets you record multiple vowels to see your complete vowel space

        You'll discover that:
        - F1 relates to tongue height (low F1 = high tongue, like [i])
        - F2 relates to tongue frontness/backness (high F2 = front tongue, like [i])
        - Everyone's vowel space is slightly different!

        *Great for: Understanding how tongue position creates different vowel sounds*
        """)

        st.markdown("---")

        st.success("""
        ### Ready to Get Started?
        👈 Choose a demo from the sidebar to begin exploring!

        **Remember:** All recordings are stored locally and will be deleted at the end
        of this session. Have fun experimenting with different sounds!
        """)

    # =============================================================================
    # DEMO 1: LIVE SPECTROGRAM & WAVEFORM
    # =============================================================================
    elif demo_mode == "📊 Live Spectrogram & Waveform":
        st.header("📊 Live Spectrogram & Waveform")
        st.markdown("""
        See your voice as a **waveform** (amplitude over time) and **spectrogram**
        (frequency content over time). Try different sounds:

        - **Vowels** - show clear harmonics (horizontal lines)
        - **[s]** sound - high-frequency noise
        - **[ʃ]** "sh" sound - lower frequency than [s]
        - **Singing** - steady pitch shows as horizontal lines
        - **Whispering** - no voicing, just noise
        """)

        duration = st.slider("Recording duration (seconds)", 2, 10, 4)

        if st.button("🎤 Record & Visualize", type="primary"):
            # Record audio
            audio, sr = record_audio(duration=duration)
            st.toast("✅ Recording complete!", icon="✅")

            # Play back audio
            st.audio(audio, sample_rate=sr)

            # Plot waveform and spectrogram
            with st.spinner("🎨 Generating visualizations..."):
                fig = plot_waveform_and_spectrogram(audio, sr)

            st.pyplot(fig)

            # Additional info
            st.info("""
            **Reading the visualizations:**
            - **Waveform (top)**: Shows how loud the sound is over time
            - **Spectrogram (bottom)**: Brighter colors = more energy at that frequency
            - Voiced sounds show regular patterns (harmonics)
            - Voiceless sounds [s, sh, f] show noise patterns
            """)

    # =============================================================================
    # DEMO 3: VOWEL PLOTTING (MULTI-VOWEL)
    # =============================================================================
    elif demo_mode == "🗣️ Vowel Plotting":
        st.header("🗣️ Vowel Plotting")
        st.markdown("""
        Have you ever wondered how changing our tonge shape can make an /i/ sound different from /a/?
        Phoneticians analyse vowel sounds by measuring their **formants** – resonant frequencies created by the shape of your vocal tract.
        The first two formants (F1 and F2) are especially important for distinguishing vowels. They tell us
        how high or low your tongue is (F1) and how front or back it is (F2).
        Record multiple vowel sounds and plot them all on a vowel chart to compare your vowel space!
        Say only the vowel, holding it steady for the duration of the recording, not the entire word.

        **Options:**
        - 🎤 **Record live** using your microphone
        - 📁 **Load demo files** from `./demo_recordings/vowels/` (if available)

        Each recording will be saved as a separate audio file. All data is stored locally in the browser and this computer—
        it will be promptly deleted at the end of this session.
        """)

        # Two-column layout
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("📝 Recording Controls")

            # Clear data button at top
            if st.button("🗑️ Clear All Data", key='clear_top', type='secondary'):
                st.session_state.vowel_recordings = []
                st.rerun()

            # IPA vowel symbols for copy-paste
            with st.expander("📋 IPA Symbols (click to copy)"):
                st.markdown("""
                **Front vowels:**
                `i` (heed)
                            
                `ɪ` (hid), 
                            
                `e` (bray), 
                            
                `ɛ` (head), 
                            
                `æ` (had)

                **Central vowels:**
                            
                `ə` ('a' in about),
                             
                `ʌ` (hut), 
                
                `ɜ` (bird)

                **Back vowels:**
                            
                `u` (who'd), 
                            
                `ʊ` (hood), 
                
                `o` (hoed), 
                            
                `ɔ` (hawed),
                            
                `ɑ` (hod)

                **Tip:** Copy the symbol above and paste into the label field!
                """)

            # Load demo recordings section
            with st.expander("📁 Load Demo Recordings"):
                demo_files = get_demo_recordings()

                if demo_files:
                    st.markdown(f"**Found {len(demo_files)} demo file(s):**")

                    for filename, filepath in demo_files:
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.write(f"📄 {filename}")
                        with col_b:
                            if st.button("📥", key=f"load_{filename}"):
                                # Load audio
                                audio, sr = load_audio_file(filepath)

                                if audio is not None:
                                    # Extract label from filename (remove extension and prefix)
                                    # Expected format: vowel_label.wav or label.wav
                                    label = filename.replace('.wav', '').replace('.WAV', '')
                                    if label.startswith('vowel_'):
                                        label = label[6:]  # Remove 'vowel_' prefix

                                    # Extract formants
                                    with st.spinner(f"🔍 Analyzing {filename}..."):
                                        formants = extract_formants(audio, sr)

                                    if 'F1' in formants and 'F2' in formants:
                                        # Add to recordings list
                                        st.session_state.vowel_recordings.append({
                                            'label': label,
                                            'f1': formants['F1'],
                                            'f2': formants['F2'],
                                            'f3': formants.get('F3', None),
                                            'audio': audio,
                                            'sample_rate': sr,
                                            'filename': filepath
                                        })
                                        st.toast(f"✅ Loaded '{label}' from demo file!", icon="✅")
                                        st.rerun()
                                    else:
                                        st.error(f"⚠️ Could not extract formants from {filename}")

                    # Load all button
                    if st.button("📥 Load All Demo Files", key='load_all_demos'):
                        loaded_count = 0
                        for filename, filepath in demo_files:
                            audio, sr = load_audio_file(filepath)
                            if audio is not None:
                                label = filename.replace('.wav', '').replace('.WAV', '')
                                if label.startswith('vowel_'):
                                    label = label[6:]

                                formants = extract_formants(audio, sr)
                                if 'F1' in formants and 'F2' in formants:
                                    st.session_state.vowel_recordings.append({
                                        'label': label,
                                        'f1': formants['F1'],
                                        'f2': formants['F2'],
                                        'f3': formants.get('F3', None),
                                        'audio': audio,
                                        'sample_rate': sr,
                                        'filename': filepath
                                    })
                                    loaded_count += 1

                        if loaded_count > 0:
                            st.toast(f"✅ Loaded {loaded_count} demo file(s)!", icon="✅")
                            st.rerun()
                else:
                    st.info("No demo files found in `./demo_recordings/vowels/`")
                    st.markdown("""
                    **To add demo files:**
                    1. Create directory: `./demo_recordings/vowels/`
                    2. Add WAV files (e.g., `vowel_i.wav`, `vowel_æ.wav`)
                    3. Refresh this page
                    """)

            st.markdown("---")
            st.markdown("**Record New Vowel:**")

            duration = st.slider("Recording duration (seconds)", 1, 5, 2, key='vowel_duration')
            vowel_label = st.text_input(
                "Vowel label (IPA symbol or word)",
                value="",
                placeholder="e.g., i, æ, or heed",
                key='vowel_label'
            )

            # Record button
            if st.button("🎤 Record Vowel", type="primary", key='record_vowel'):
                if not vowel_label:
                    st.warning("⚠️ Please enter a label for your vowel!")
                else:
                    # Record audio
                    audio, sr = record_audio(duration=duration)
                    st.toast("✅ Recording complete!", icon="✅")

                    # Play back audio
                    st.audio(audio, sample_rate=sr)

                    # Extract formants
                    with st.spinner("🔍 Analyzing formants..."):
                        formants = extract_formants(audio, sr)

                    # Save audio file
                    with st.spinner("💾 Saving audio file..."):
                        filename = save_audio_file(audio, sr, vowel_label)

                    # Display formant values
                    if 'F1' in formants and 'F2' in formants:
                        st.success(f"📁 Saved as: `{filename}`")
                        st.metric("F1 (First Formant)", f"{formants['F1']:.0f} Hz")
                        st.metric("F2 (Second Formant)", f"{formants['F2']:.0f} Hz")
                        if 'F3' in formants:
                            st.metric("F3 (Third Formant)", f"{formants['F3']:.0f} Hz")

                        # Add to recordings list
                        st.session_state.vowel_recordings.append({
                            'label': vowel_label,
                            'f1': formants['F1'],
                            'f2': formants['F2'],
                            'f3': formants.get('F3', None),
                            'audio': audio,
                            'sample_rate': sr,
                            'filename': filename
                        })

                        st.toast(f"Added '{vowel_label}' to your vowel chart!")
                    else:
                        st.warning("⚠️ Could not extract formants. Try speaking louder or holding the vowel longer.")

            # Display recorded vowels list
            st.markdown("---")
            st.subheader(f"📊 Recorded Vowels ({len(st.session_state.vowel_recordings)})")

            if st.session_state.vowel_recordings:
                for idx, vowel in enumerate(st.session_state.vowel_recordings):
                    with st.container():
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.write(f"**{idx+1}. {vowel['label']}** - F1: {vowel['f1']:.0f} Hz, F2: {vowel['f2']:.0f} Hz")
                        with col_b:
                            if st.button("🗑️", key=f"delete_{idx}"):
                                st.session_state.vowel_recordings.pop(idx)
                                st.rerun()

                # Clear all button
                if st.button("🗑️ Clear All", key='clear_all'):
                    st.session_state.vowel_recordings = []
                    st.rerun()
            else:
                st.info("No vowels recorded yet. Record some vowels to get started!")

        with col2:
            st.subheader("📈 Vowel Space Chart")

            # Plot vowel chart
            if st.session_state.vowel_recordings:
                fig = plot_vowel_chart_multi(st.session_state.vowel_recordings)
                st.pyplot(fig)

                # Download button for the chart
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                st.download_button(
                    label="📥 Download Chart",
                    data=buf,
                    file_name="vowel_chart.png",
                    mime="image/png"
                )
            else:
                # Show empty chart with reference vowels
                fig = plot_vowel_chart_multi(None)
                st.pyplot(fig)
                st.info("👆 Record vowels to see them plotted on the chart!")

    # =============================================================================
    # DEMO 2: PITCH CONTOUR / INTONATION (MULTI-RECORDING)
    # =============================================================================
    elif demo_mode == "❓ Is This a Question?":
        st.header("❓ Is This a Question?")
        st.markdown("""
        Have you ever wondered how we can tell if someone is asking a question just by the way their voice goes up at the end?
        This demo lets you explore the magic of **intonation** – how pitch changes over the course of an utterance to convey meaning.
        
        Record up to 3 utterances and compare their pitch contours side-by-side!
        Great for comparing statements vs. questions, or different intonation patterns.

        **Options:**
        - 🎤 **Record live** (up to 3 recordings)
        - 📁 **Load demo files** from `./demo_recordings/intonation/` (if available)

        **Try saying:**
        - "You're going to the store" (statement) ⬇️
        - "You're going to the store?" (question) ⬆️
        """)

        # Two-column layout
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("📝 Recording Controls")

            # Clear data button at top
            if st.button("🗑️ Clear All Data", key='clear_pitch_top', type='secondary'):
                st.session_state.pitch_recordings = []
                st.rerun()

            # Load demo recordings section
            with st.expander("📁 Load Demo Recordings"):
                demo_dir = "./demo_recordings/intonation"
                demo_files = get_demo_recordings() if os.path.exists(demo_dir) else []

                # Modify get_demo_recordings to use intonation folder
                intonation_files = []
                if os.path.exists(demo_dir):
                    for filename in os.listdir(demo_dir):
                        if filename.endswith('.wav') or filename.endswith('.WAV'):
                            full_path = os.path.join(demo_dir, filename)
                            intonation_files.append((filename, full_path))
                    intonation_files = sorted(intonation_files)

                if intonation_files:
                    st.markdown(f"**Found {len(intonation_files)} demo file(s):**")

                    for filename, filepath in intonation_files:
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.write(f"📄 {filename}")
                        with col_b:
                            if st.button("📥", key=f"load_pitch_{filename}"):
                                # Check if we've hit the limit
                                if len(st.session_state.pitch_recordings) >= 3:
                                    st.warning("⚠️ Maximum 3 recordings. Delete one to add more.")
                                else:
                                    # Load audio
                                    audio, sr = load_audio_file(filepath)

                                    if audio is not None:
                                        # Extract label from filename
                                        label = filename.replace('.wav', '').replace('.WAV', '')

                                        # Extract pitch
                                        with st.spinner(f"🔍 Analyzing {filename}..."):
                                            time_points, pitch_values = extract_pitch(audio, sr)

                                        if len(pitch_values) > 0:
                                            # Add to recordings list
                                            st.session_state.pitch_recordings.append({
                                                'label': label,
                                                'time': time_points,
                                                'pitch': pitch_values,
                                                'audio': audio,
                                                'sample_rate': sr,
                                                'filename': filepath
                                            })
                                            st.toast(f"✅ Loaded '{label}' from demo file!", icon="✅")
                                            st.rerun()
                                        else:
                                            st.error(f"⚠️ Could not extract pitch from {filename}")
                else:
                    st.info("No demo files found in `./demo_recordings/intonation/`")
                    st.markdown("""
                    **To add demo files:**
                    1. Create directory: `./demo_recordings/intonation/`
                    2. Add WAV files (e.g., `statement.wav`, `question.wav`)
                    3. Refresh this page
                    """)

            st.markdown("---")
            st.markdown("**Record New Utterance:**")

            duration = st.slider("Recording duration (seconds)", 2, 6, 3, key='pitch_duration')
            utterance_label = st.text_input(
                "Label (e.g., 'statement', 'question')",
                value="",
                placeholder="e.g., statement, question",
                key='utterance_label'
            )

            # Check if we've hit the recording limit
            can_record = len(st.session_state.pitch_recordings) < 3

            if not can_record:
                st.warning("⚠️ Maximum 3 recordings reached. Delete one to record more.")

            # Record button
            if st.button("🎤 Record Utterance", type="primary", key='record_pitch', disabled=not can_record):
                if not utterance_label:
                    st.warning("⚠️ Please enter a label for your utterance!")
                else:
                    # Record audio
                    audio, sr = record_audio(duration=duration)
                    st.toast("✅ Recording complete!", icon="✅")

                    # Play back audio
                    st.audio(audio, sample_rate=sr)

                    # Extract pitch
                    with st.spinner("🔍 Analyzing pitch contour..."):
                        time_points, pitch_values = extract_pitch(audio, sr)

                    if len(pitch_values) > 0:
                        # Save audio file
                        os.makedirs('recordings/intonation', exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        safe_label = utterance_label.replace(' ', '_').replace('/', '_')
                        filename = f"recordings/intonation/{safe_label}_{timestamp}.wav"
                        sf.write(filename, audio, sr)

                        st.success(f"📁 Saved as: `{filename}`")

                        # Display pitch statistics
                        st.subheader("Pitch Statistics")
                        st.metric("Mean Pitch", f"{np.mean(pitch_values):.1f} Hz")
                        st.metric("Pitch Range", f"{np.max(pitch_values) - np.min(pitch_values):.1f} Hz")

                        # Determine pattern
                        if len(pitch_values) > 2:
                            third = len(pitch_values) // 3
                            start_avg = np.mean(pitch_values[:third])
                            end_avg = np.mean(pitch_values[-third:])

                            if end_avg > start_avg * 1.1:
                                pattern = "Rising ⬆️"
                            elif end_avg < start_avg * 0.9:
                                pattern = "Falling ⬇️"
                            else:
                                pattern = "Level ➡️"

                            st.info(f"Pattern: **{pattern}**")

                        # Add to recordings list
                        st.session_state.pitch_recordings.append({
                            'label': utterance_label,
                            'time': time_points,
                            'pitch': pitch_values,
                            'audio': audio,
                            'sample_rate': sr,
                            'filename': filename
                        })

                        st.toast(f"Added '{utterance_label}' to comparison chart!")
                    else:
                        st.warning("⚠️ Could not extract pitch. Try speaking louder.")

            # Display recorded utterances list
            st.markdown("---")
            st.subheader(f"📊 Recorded Utterances ({len(st.session_state.pitch_recordings)}/3)")

            if st.session_state.pitch_recordings:
                for idx, recording in enumerate(st.session_state.pitch_recordings):
                    with st.container():
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            mean_pitch = np.mean(recording['pitch'])
                            st.write(f"**{idx+1}. {recording['label']}** - Mean: {mean_pitch:.1f} Hz")
                            # Add audio player for this recording
                            st.audio(recording['audio'], sample_rate=recording['sample_rate'])
                        with col_b:
                            if st.button("🗑️", key=f"delete_pitch_{idx}"):
                                st.session_state.pitch_recordings.pop(idx)
                                st.rerun()

                # Clear all button
                if st.button("🗑️ Clear All", key='clear_pitch_all'):
                    st.session_state.pitch_recordings = []
                    st.rerun()
            else:
                st.info("No utterances recorded yet. Record up to 3 to compare!")

        with col2:
            st.subheader("📈 Pitch Contour Comparison")

            # Show tips
            st.info("""
            **What to look for:**
            - 📈 **Rising** pitch at the end = Question
            - 📉 **Falling** pitch at the end = Statement
            - ➡️ **Level** pitch = Flat/monotone

            **Line markers (colorblind accessible):**
            - ■ Square = First recording (blue)
            - ▲ Triangle = Second recording (red)
            - ● Circle = Third recording (green)
            """)

            # Plot pitch contours
            if st.session_state.pitch_recordings:
                fig = plot_pitch_contours_multi(st.session_state.pitch_recordings)
                st.pyplot(fig)

                # Download button for the chart
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                st.download_button(
                    label="📥 Download Chart",
                    data=buf,
                    file_name="pitch_comparison.png",
                    mime="image/png"
                )

                # Add spectrograms with overlaid pitch contours
                st.markdown("---")
                st.subheader("🎵 Spectrograms with Pitch Overlay")
                st.markdown("*Each recording shown as a spectrogram with its pitch contour overlaid*")

                fig_spectro = plot_spectrograms_with_pitch(st.session_state.pitch_recordings)
                if fig_spectro:
                    st.pyplot(fig_spectro)

                    # Download button for spectrograms
                    buf_spectro = io.BytesIO()
                    fig_spectro.savefig(buf_spectro, format='png', dpi=300, bbox_inches='tight')
                    buf_spectro.seek(0)
                    st.download_button(
                        label="📥 Download Spectrograms",
                        data=buf_spectro,
                        file_name="spectrograms_with_pitch.png",
                        mime="image/png",
                        key="download_spectro"
                    )
            else:
                # Show empty chart
                fig = plot_pitch_contours_multi(None)
                st.pyplot(fig)
                st.info("👆 Record or load utterances to see pitch contours!")


if __name__ == "__main__":
    main()
