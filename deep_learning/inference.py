from pathlib import Path

import torch
import librosa
import soundfile as sf
import numpy as np
from model import NeuralAmp

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Procesando audio en: {device}")

    # 1. Cargar el modelo de forma segura (sin avisos de seguridad)
    model = NeuralAmp(hidden_size=64, num_layers=2).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "mi_ampli_entrenado.pth", map_location=device, weights_only=True))
    model.eval()

    # 2. Cargar la guitarra limpia
    input_path = DATA_DIR / "guitarra_clean.wav"
    sample_rate = 44100
    x, _ = librosa.load(str(input_path), sr=sample_rate, mono=True)

    # Normalización
    x_norm = x / np.max(np.abs(x))

    # 3. Procesar por bloques para no saturar cuDNN (Bloques de 4096 muestras ~ 90ms)
    chunk_size = 4096
    output_chunks = []

    print("Pasando el audio por el amplificador virtual por bloques...")

    # Inicializamos los estados ocultos de la LSTM en None
    # Al mantener el estado entre bloques, la LSTM no pierde la 'memoria' del sonido anterior
    hidden = None

    with torch.no_grad():
        for i in range(0, len(x_norm), chunk_size):
            chunk = x_norm[i:i+chunk_size]

            # Si el último bloque es más corto, lo rellenamos con ceros (padding)
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')

            # Convertir a tensor, forzar orden contiguo en memoria y pasar a GPU
            chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            chunk_tensor = chunk_tensor.contiguous().to(device)

            # En la primera iteración se inicializan los estados ocultos de la LSTM automáticamente
            if hidden is None:
                # model.lstm espera la entrada. Si pasamos el bloque directamente, extraemos la lógica
                lstm_out, hidden = model.lstm(chunk_tensor)
            else:
                # Pasamos el estado oculto anterior para dar continuidad al audio
                lstm_out, hidden = model.lstm(chunk_tensor, hidden)

            # Pasamos el output por la capa lineal final del modelo
            pred_chunk = model.linear(lstm_out)

            # Guardamos el audio devuelto eliminando las dimensiones extra
            pred_chunk_np = pred_chunk.squeeze().cpu().numpy()
            output_chunks.append(pred_chunk_np)

    # 4. Concatenar todos los bloques procesados
    output_audio = np.concatenate(output_chunks)[:len(x_norm)]

    # Desnormalizar un poco para evitar clipping duro digital en el WAV
    output_audio = output_audio / np.max(np.abs(output_audio)) * 0.9

    # 5. Guardar el resultado
    output_path = DATA_DIR / "guitarra_predicha_IA.wav"
    sf.write(str(output_path), output_audio, sample_rate)
    print(f"¡Audio procesado con éxito! Guardado en: {output_path}")


if __name__ == "__main__":
    run_inference()
