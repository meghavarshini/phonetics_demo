
import streamlit as st
import numpy as np
import parselmouth
from parselmouth.praat import call
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from scipy.io import wavfile
import io
import tempfile
import os

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

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def record_audio(duration=3, sample_rate=44100):
    """Record audio from the microphone.
    
    Args:
        duration (int): Recording duration in seconds
        sample_rate (int): Sampling rate in Hz
    
    Returns:
        tuple: (audio_data, sample_rate)
    """
    with st.spinner(f"🎤 Recording for {duration} seconds..."):
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float64'
        )
        sd.wait()  # Wait until recording is finished
    return audio.flatten(), sample_rate


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
    """Extract pitch contour using Parselmouth/Praat.
    
    Args:
        audio (np.array): Audio signal
        sample_rate (int): Sampling rate
    
    Returns:
        tuple: (time_points, pitch_values)
    """
    sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)
    
    # Extract pitch using autocorrelation (Praat default)
    pitch = call(sound, "To Pitch", 0.0, 75, 600)  # 75-600 Hz range for human speech
    
    # Get pitch values over time
    time_points = []
    pitch_values = []
    
    for t in np.arange(0, sound.duration, 0.01):  # Sample every 10ms
        pitch_value = call(pitch, "Get value at time", t, "Hertz", "Linear")
        if not np.isnan(pitch_value) and pitch_value > 0:
            time_points.append(t)
            pitch_values.append(pitch_value)
    
    return np.array(time_points), np.array(pitch_values)


def plot_vowel_chart(f1, f2, vowel_label=None):
    """Plot a vowel on a standard F1-F2 vowel chart.
    
    Args:
        f1 (float): First formant frequency
        f2 (float): Second formant frequency
        vowel_label (str): Label for the vowel point
    
    Returns:
        matplotlib.figure.Figure: The vowel chart figure
    """
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    
    # Reference vowels (approximate American English values)
    reference_vowels = {
        'i (heed)': (280, 2250),
        'ɪ (hid)': (400, 1900),
        'e (hayed)': (400, 2100),
        'ɛ (head)': (550, 1800),
        'æ (had)': (700, 1700),
        'ɑ (hod)': (700, 1100),
        'ɔ (hawed)': (600, 900),
        'o (hoed)': (450, 850),
        'ʊ (hood)': (400, 1000),
        'u (who\'d)': (300, 900)
    }
    
    # Plot reference vowels
    for vowel, (f1_ref, f2_ref) in reference_vowels.items():
        ax.scatter(f2_ref, f1_ref, s=200, alpha=0.3, c=COLORS['secondary'], 
                  edgecolors=COLORS['dark'], linewidth=2)
        ax.annotate(vowel, (f2_ref, f1_ref), fontsize=11, ha='center', 
                   va='center', weight='bold')
    
    # Plot user's vowel
    if f1 and f2:
        ax.scatter(f2, f1, s=500, c=COLORS['primary'], marker='*', 
                  edgecolors=COLORS['dark'], linewidth=3, zorder=10, 
                  label='Your vowel')
        if vowel_label:
            ax.annotate(vowel_label, (f2, f1), xytext=(0, 30), 
                       textcoords='offset points', fontsize=14, 
                       weight='bold', color=COLORS['primary'],
                       ha='center',
                       bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor=COLORS['accent'], alpha=0.7))
    
    # Invert axes (phonetic convention)
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    # Labels and formatting
    ax.set_xlabel('F2 (Hz)', fontsize=14, weight='bold')
    ax.set_ylabel('F1 (Hz)', fontsize=14, weight='bold')
    ax.set_title('Vowel Space Chart (F1 vs F2)', fontsize=16, weight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if f1 and f2:
        ax.legend(fontsize=12, loc='upper right')
    
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
    ax.legend(fontsize=12)
    
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
        ["🗣️ Vowel Plotting", "❓ Is This a Question?", "📊 Live Spectrogram & Waveform"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Tips:**
    - Speak clearly into your microphone
    - Find a quiet environment
    - Try different sounds and see what happens!
    """)
    
    # =============================================================================
    # DEMO 1: VOWEL PLOTTING
    # =============================================================================
    if demo_mode == "🗣️ Vowel Plotting":
        st.header("🗣️ Vowel Plotting")
        st.markdown("""
        Record yourself saying a vowel sound, and we'll plot its **formants** (resonant frequencies) 
        on a vowel chart. Different vowels occupy different regions of the chart!
        
        **Try these vowels:**
        - "ee" as in *heed*
        - "aa" as in *had*
        - "oo" as in *who'd*
        - "ah" as in *hod*
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Recording Settings")
            duration = st.slider("Recording duration (seconds)", 1, 5, 2)
            vowel_label = st.text_input("Label your vowel (optional)", "")
            
            if st.button("🎤 Record Vowel", type="primary"):
                # Record audio
                audio, sr = record_audio(duration=duration)
                st.success("✅ Recording complete!")
                
                # Play back audio
                st.audio(audio, sample_rate=sr)
                
                # Extract formants
                with st.spinner("🔍 Analyzing formants..."):
                    formants = extract_formants(audio, sr)
                
                # Display formant values
                st.subheader("Formant Frequencies")
                if 'F1' in formants and 'F2' in formants:
                    st.metric("F1 (First Formant)", f"{formants['F1']:.0f} Hz")
                    st.metric("F2 (Second Formant)", f"{formants['F2']:.0f} Hz")
                    if 'F3' in formants:
                        st.metric("F3 (Third Formant)", f"{formants['F3']:.0f} Hz")
                    
                    # Store in session state for plotting
                    st.session_state['f1'] = formants['F1']
                    st.session_state['f2'] = formants['F2']
                    st.session_state['vowel_label'] = vowel_label if vowel_label else "Your vowel"
                else:
                    st.warning("⚠️ Could not extract formants. Try speaking louder or closer to the mic.")
        
        with col2:
            st.subheader("Vowel Chart")
            # Plot vowel chart
            if 'f1' in st.session_state and 'f2' in st.session_state:
                fig = plot_vowel_chart(
                    st.session_state['f1'],
                    st.session_state['f2'],
                    st.session_state.get('vowel_label', 'Your vowel')
                )
                st.pyplot(fig)
            else:
                # Show empty chart
                fig = plot_vowel_chart(None, None)
                st.pyplot(fig)
                st.info("👆 Record a vowel to see where it appears on the chart!")
    
    # =============================================================================
    # DEMO 2: PITCH CONTOUR / INTONATION
    # =============================================================================
    elif demo_mode == "❓ Is This a Question?":
        st.header("❓ Is This a Question?")
        st.markdown("""
        Say the same sentence two ways - as a **statement** (falling pitch) and as a 
        **question** (rising pitch). See how the pitch contour changes!
        
        **Try saying:**
        - "You're going to the store" (statement) ⬇️
        - "You're going to the store?" (question) ⬆️
        """)
        
        duration = st.slider("Recording duration (seconds)", 2, 6, 3)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎤 Record Utterance", type="primary"):
                # Record audio
                audio, sr = record_audio(duration=duration)
                st.success("✅ Recording complete!")
                
                # Play back audio
                st.audio(audio, sample_rate=sr)
                
                # Extract pitch
                with st.spinner("🔍 Analyzing pitch contour..."):
                    time_points, pitch_values = extract_pitch(audio, sr)
                
                if len(pitch_values) > 0:
                    # Store in session state
                    st.session_state['pitch_time'] = time_points
                    st.session_state['pitch_values'] = pitch_values
                    
                    # Display pitch statistics
                    st.subheader("Pitch Statistics")
                    st.metric("Mean Pitch", f"{np.mean(pitch_values):.1f} Hz")
                    st.metric("Pitch Range", f"{np.max(pitch_values) - np.min(pitch_values):.1f} Hz")
                else:
                    st.warning("⚠️ Could not extract pitch. Try speaking louder.")
        
        with col2:
            # Show instructions or stats
            st.info("""
            **What to look for:**
            - 📈 **Rising** pitch at the end = Question
            - 📉 **Falling** pitch at the end = Statement
            - ➡️ **Level** pitch = Flat/monotone
            """)
        
        # Plot pitch contour
        if 'pitch_time' in st.session_state and 'pitch_values' in st.session_state:
            st.subheader("Pitch Contour")
            fig = plot_pitch_contour(
                st.session_state['pitch_time'],
                st.session_state['pitch_values']
            )
            st.pyplot(fig)
        else:
            st.info("👆 Record your voice to see the pitch contour!")
    
    # =============================================================================
    # DEMO 3: LIVE SPECTROGRAM & WAVEFORM
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
            st.success("✅ Recording complete!")
            
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


if __name__ == "__main__":
    main()
