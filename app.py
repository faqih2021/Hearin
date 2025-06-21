import streamlit as st
import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import os
import soundfile as sf
import io
import base64

APP_NAME = "Hearin"
APP_DESCRIPTION = "Mengubah Suara Menjadi Teks untuk Membantu Orang Sekitarmu"
LOGO_PATH = "Hearin.png" 
ORANGE_COLOR = "#FF6347"

# Pengaturan Model STT
MODEL_NAME = "openai/whisper-tiny"

MODEL_LOCAL_DIR = "./models/stt_model"

@st.cache_resource
def load_stt_model():
    try:
        # st.info(f"Menginisialisasi pipeline dengan model: '{MODEL_NAME}'...")
        
        stt_pipeline = pipeline(
            "automatic-speech-recognition",
            model=MODEL_NAME,
            device=0 if torch.cuda.is_available() else -1,
        )
        
        if not os.path.exists(MODEL_LOCAL_DIR) or not os.path.isdir(MODEL_LOCAL_DIR):
            st.info(f"Mengunduh dan menyimpan komponen model ke: {MODEL_LOCAL_DIR}")
            model_obj = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME)
            processor_obj = AutoProcessor.from_pretrained(MODEL_NAME)
            model_obj.save_pretrained(MODEL_LOCAL_DIR)
            processor_obj.save_pretrained(MODEL_LOCAL_DIR)
        
        st.success("Model Speech-to-Text berhasil dimuat!")
        return stt_pipeline
    except Exception as e:
        st.error(f"Gagal memuat model STT: {e}")
        st.warning("Pastikan Anda memiliki koneksi internet saat pertama kali mengunduh model, atau pastikan model sudah ada di folder 'models/stt_model'.")
        st.warning(f"Coba ganti MODEL_NAME dengan model yang valid, misalnya 'openai/whisper-tiny' atau 'indonesian-nlp/wav2vec2-large-xlsr-indonesian'.")
        return None

# Model loader
stt_model = load_stt_model()


# Streamlit 
st.set_page_config(
    page_title=APP_NAME,
    page_icon=LOGO_PATH,
    layout="centered",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown(f"""
    <style>
        .stApp {{
            background-color: #f0f2f6;
        }}
        .reportview-container .main .block-container {{
            padding-top: 2rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 2rem;
        }}
        .stButton>button {{
            background-color: {ORANGE_COLOR};
            color: white;
            border-radius: 5px;
            border: none;
            padding: 0.5rem 1rem;
            font-size: 1rem;
            font-weight: bold;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        }}
        .stButton>button:hover {{
            background-color: #e65230;
            color: white;
        }}
        .stFileUploader {{
            border: 2px dashed {ORANGE_COLOR};
            padding: 1rem;
            border-radius: 10px;
            background-color: #fff;
        }}
        .stFileUploader label {{
            color: {ORANGE_COLOR};
            font-weight: bold;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {ORANGE_COLOR};
            text-align: center;
        }}
        .stMarkdown p {{
            text-align: center;
            color: #555;
        }}
        /* Perbaikan CSS untuk sidebar */
        .css-1d391kg {{ /* Untuk sidebar header */
            background-color: {ORANGE_COLOR};
            color: white;
        }}
        .css-1lcbmhc {{ /* Untuk sidebar content */
            background-color: #f7f7f7;
        }}
        /* CSS khusus untuk gambar di sidebar */
        .sidebar-logo-container {{
            width: 100%;
            text-align: center;
            margin-bottom: 15px;
        }}
        .sidebar-logo-container img {{
            display: inline-block;
            max-width: 120px; /* Lebar maksimum logo, bisa disesuaikan */
            height: auto;
        }}
    </style>
""", unsafe_allow_html=True)

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
# st.image(LOGO_PATH, width=150)
 st.markdown(f"<h1 style='text-align: center; color: {ORANGE_COLOR};'>{APP_NAME}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center;'>{APP_DESCRIPTION}</p>", unsafe_allow_html=True)
st.markdown("---")

# Upload
st.subheader("Unggah File Audio Anda")
uploaded_file = st.file_uploader("Pilih file audio (.wav, .mp3, .ogg)", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file, format=uploaded_file.type)

    if st.button("Ubah Suara Menjadi Teks"):
        if stt_model is not None:
            with st.spinner("Memproses audio..."):
                try:
                    temp_audio_path = "temp_audio.wav"
                    with open(temp_audio_path, "wb") as f:
                        f.write(uploaded_file.read())

                    transcription = stt_model(temp_audio_path)
                    transcribed_text = transcription["text"]
                    
                    transcribed_text_formatted = transcribed_text.capitalize() 

                    st.markdown(f"## Hasil Transkripsi:")
                    st.success(f"**{transcribed_text_formatted}**") 
                    
                    download_filename = "transkripsi_hearin.txt"
                    st.download_button(
                        label="Download Transkripsi (.txt)",
                        data=transcribed_text_formatted,
                        file_name=download_filename,
                        mime="text/plain"
                    )

                    os.remove(temp_audio_path)

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses audio: {e}")
        else:
            st.warning("Model Speech-to-Text belum dimuat.")

st.markdown("---")
st.info("Catatan: Akurasi transkripsi dapat bervariasi tergantung kualitas audio dan model yang digunakan.")

# Sidebar 
with st.sidebar:
    st.markdown(f"<h2 style='color: black;'>Tentang {APP_NAME}</h2>", unsafe_allow_html=True)
    st.markdown("""
    **Hearin** adalah aplikasi yang dirancang untuk **membantu individu tunarungu**
    agar dapat **memahami apa yang diucapkan** di lingkungan sekitar mereka.
    Aplikasi ini bekerja dengan mengubah suara dari file audio menjadi teks,
    memungkinkan pengguna untuk membaca dan mengikuti percakapan.
    """)

    try:
        with open(LOGO_PATH, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        
        st.markdown(f"""
            <div class="sidebar-logo-container">
                <img src='data:image/png;base64,{encoded_string}' alt='Hearin Logo'>
            </div>
            """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Logo '{LOGO_PATH}' tidak ditemukan.")
    except Exception as e:
        st.error(f"Gagal memuat logo: {e}")

    st.markdown("---")
    st.markdown(f"""
    **Cara Kerja:**
    Hearin menggunakan teknologi Neural Network (Speech-to-Text) untuk
    menganalisis pola suara dan mengubahnya menjadi rangkaian kata-kata.
    Aplikasi ini dapat memproses file audio yang Anda unggah.
    """)