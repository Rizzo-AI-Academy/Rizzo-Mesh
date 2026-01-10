import asyncio
import os
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# 1. AUTHENTICATION & CONFIGURATION
# Cerchiamo la chiave API. Se l'utente usa SDK "Agents" e non la mette, 
# potrebbe affidarsi a "Google Application Default Credentials" (ADC).
# Quindi configuriamo l'API key SOLO se presente, altrimenti lasciamo che l'SDK faccia il suo dovere.
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    print("⚠ Nessuna API Key trovata in .env. Si assume autenticazione tramite ADC (gcloud auth) o variabili di sistema.")

# 2. PROMPT
system_prompt = """
Agisci come un esperto di analisi spaziale 3D. L'immagine fornita contiene diverse angolazioni dello STESSO oggetto. Istruzioni: 1. Combina mentalmente le viste per ricostruire la geometria completa dell'oggetto. 2. Identifica l'oggetto basandoti esclusivamente sulla morfologia. 3. Restituisci SOLO una descrizione sintetica dell'oggetto identificato. Vincoli rigorosi: - Non menzionare assolutamente il colore, i materiali o lo sfondo. - Non usare frasi introduttive (es. "Questo è...", "Vedo..."). - Lunghezza massima: 10-15 parole.
"""

# 3. GLOBAL MODEL INITIALIZATION (Persistent)
# Inizializziamo il modello UNA sola volta per evitare overhead/reset della connessione
try:
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    
    generation_config = genai.types.GenerationConfig(
        max_output_tokens=1024, # Abbondiamo per evitare tagli
        temperature=0.7,
        top_p=0.95
    )
    
    # Usa gemini-3 se disponibile, altrimenti fallback automatici
    model_name = "models/gemini-3-flash-preview"
    print(f"🔹 Initializing Global Model: {model_name}")
    model = genai.GenerativeModel(model_name=model_name, 
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)

except Exception as e:
    print(f"⚠ Errore inizializzazione modello Globale: {e}")
    model = None

# 4. ASYNC FUNCTION
async def ai_captioning(img_path):
    """
    Funzione asincrona per chiamare l'API Gemini e ottenere la caption.
    Usa l'istanza globale 'model'.
    """
    if not model:
        print("❌ Modello non inizializzato.")
        return None

    try:
        if not os.path.exists(img_path):
            print(f"❌ Immagine non trovata: {img_path}")
            return None

        # Carica immagine (PIL Image object)
        img = Image.open(img_path)
        
        # RATE LIMITING: Piccola pausa per evitare di bombardare l'API e andare in quota error
        await asyncio.sleep(1.0)
        
        # CHIAMATA ASINCRONA CON TIMEOUT
        # Se Gemini non risponde entro 30 secondi, abortiamo questa specifica richiesta
        try:
            response = await asyncio.wait_for(
                model.generate_content_async(
                    contents=[system_prompt, img]
                ),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            print(f"⏰ TIMEOUT Gemini per {os.path.basename(img_path)} (skip)")
            return None
        
        # Debug Risposta
        if response.candidates and response.candidates[0].finish_reason != 1:
             print(f"⚠ Finish Reason anomalous: {response.candidates[0].finish_reason} for {os.path.basename(img_path)}")

        if response and response.text:
            return response.text.strip()
        else:
            return None
            
    except Exception as e:
        print(f"❌ Gemini Error ({os.path.basename(img_path)}): {e}")
        return None