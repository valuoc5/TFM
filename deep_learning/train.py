from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

from model import NeuralAmp

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# ==========================================
# 0. CONFIGURACIÓN
# ==========================================
INPUT_FILE = "guitarra_clean_2.wav"
TARGET_FILE = "guitarra_dist_2.wav"
MODEL_FILE = "mi_ampli_entrenado_2.pth"

SAMPLE_RATE = 44100
# Las pistas están en estéreo y el canal izquierdo está amplificado y saturado
# en la pista limpia. El canal derecho (1) es una copia limpia, así que usamos ese.
CHANNEL = 1

# El plugin retrasa la salida (~45 muestras) porque sus filtros internos miran un
# poco "hacia delante". Si compensamos todo el retardo, la red tendría que adivinar
# el futuro. Dejamos un margen sin compensar para que solo necesite el pasado.
MARGEN_RETARDO = 25

# Primeras muestras de cada secuencia que no cuentan en la pérdida (warm-up):
# la LSTM empieza cada secuencia sin memoria y necesita unas muestras para "arrancar".
WARMUP = 256

# Filtro de pre-énfasis en la pérdida (Wright y Välimäki, 2020): da más peso a los
# agudos para que la red no genere el siseo de alta frecuencia que el oído sí nota.
PRE_ENFASIS = 0.85

# Límite del tamaño de los gradientes, para evitar saltos bruscos del entrenamiento
MAX_GRAD_NORM = 1.0

# División temporal del audio: 70 % entrenamiento, 15 % validación, 15 % test
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15


# ==========================================
# 1. CARGA Y PREPARACIÓN DE LOS DATOS
# ==========================================
def cargar_canal(path):
    """Carga un WAV y devuelve un único canal como array 1D."""
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=False)
    if audio.ndim == 2:
        audio = audio[CHANNEL]
    return audio


def estimar_retardo(x, y, max_lag=2000, segundos=60):
    """Retardo (en muestras) de y respecto a x, por correlación cruzada.

    Se usa el valor absoluto porque el plugin invierte la polaridad.
    """
    n = min(len(x), len(y), segundos * SAMPLE_RATE)
    a, b = y[:n], x[:n]
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    corr = np.fft.irfft(np.fft.rfft(a, nfft) * np.conj(np.fft.rfft(b, nfft)), nfft)
    corr = np.concatenate([corr[-max_lag:], corr[:max_lag + 1]])
    lags = np.arange(-max_lag, max_lag + 1)
    return int(lags[np.argmax(np.abs(corr))])


def preparar_datos(input_path, target_path):
    x = cargar_canal(input_path)
    y = cargar_canal(target_path)

    # Sincronizar longitudes exactas
    min_len = min(len(x), len(y))
    x, y = x[:min_len], y[:min_len]

    # Compensar el retardo del plugin, dejando un margen (ver MARGEN_RETARDO)
    lag_medido = estimar_retardo(x, y)
    lag = max(lag_medido - MARGEN_RETARDO, 0) if lag_medido > 0 else lag_medido
    print(f"Retardo medido: {lag_medido} muestras")
    if lag > 0:
        x, y = x[:-lag], y[lag:]
    elif lag < 0:
        x, y = x[-lag:], y[:lag]
    print(f"Retardo compensado: {lag} muestras ({1000 * lag / SAMPLE_RATE:.2f} ms)")

    # Normalizar amplitud entre -1.0 y 1.0
    x = x / np.max(np.abs(x))
    y = y / np.max(np.abs(y))

    # División temporal en entrenamiento, validación y test
    n = len(x)
    i_train = int(n * TRAIN_FRAC)
    i_val = int(n * (TRAIN_FRAC + VAL_FRAC))
    partes = {
        "train": (x[:i_train], y[:i_train]),
        "val": (x[i_train:i_val], y[i_train:i_val]),
        "test": (x[i_val:], y[i_val:]),
    }
    for nombre, (xs, _) in partes.items():
        print(f"  {nombre}: {len(xs) / SAMPLE_RATE:.1f} s")
    return partes


class GuitarAmpDataset(Dataset):
    def __init__(self, x, y, sequence_length=1024):
        self.x = x
        self.y = y
        self.sequence_length = sequence_length
        self.num_samples = len(x) // sequence_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.sequence_length
        end = start + self.sequence_length

        x_chunk = self.x[start:end]
        y_chunk = self.y[start:end]

        # Formato esperado por LSTM: (Secuencia, Canales) -> (1024, 1)
        x_tensor = torch.tensor(x_chunk, dtype=torch.float32).unsqueeze(-1)
        y_tensor = torch.tensor(y_chunk, dtype=torch.float32).unsqueeze(-1)

        return x_tensor, y_tensor


# ==========================================
# 2. LA FUNCIÓN DE PÉRDIDA (ESR Loss)
# ==========================================
class ESRLoss(nn.Module):
    def __init__(self, epsilon=1e-7, pre_enfasis=0.0):
        super(ESRLoss, self).__init__()
        self.epsilon = epsilon
        self.pre_enfasis = pre_enfasis

    def filtrar(self, y):
        # Filtro paso alto de primer orden: y[n] - a * y[n-1]
        return y[:, 1:, :] - self.pre_enfasis * y[:, :-1, :]

    def forward(self, y_pred, y_true):
        if self.pre_enfasis > 0:
            y_pred, y_true = self.filtrar(y_pred), self.filtrar(y_true)
        numerator = torch.sum((y_true - y_pred) ** 2)
        denominator = torch.sum(y_true ** 2) + self.epsilon
        return numerator / denominator


# ==========================================
# 3. EVALUACIÓN SOBRE AUDIO CONTINUO
# ==========================================
def predecir(model, x, device, chunk_size=44100):
    """Procesa un audio completo por bloques, manteniendo el estado de la LSTM."""
    model.eval()
    hidden = None
    salida = []
    with torch.no_grad():
        for i in range(0, len(x), chunk_size):
            chunk = torch.tensor(x[i:i + chunk_size], dtype=torch.float32)
            chunk = chunk.view(1, -1, 1).to(device)
            lstm_out, hidden = model.lstm(chunk, hidden)
            salida.append(model.linear(lstm_out).view(-1).cpu().numpy())
    return np.concatenate(salida)


def esr(y_true, y_pred):
    return float(np.sum((y_true - y_pred) ** 2) / (np.sum(y_true ** 2) + 1e-7))


# ==========================================
# 4. EL BUCLE DE ENTRENAMIENTO
# ==========================================
def train_model():
    # Forzar uso de GPU CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en: {device}")

    # Hiperparámetros base
    EPOCHS = 200
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    print("Cargando archivos de audio en memoria...")
    datos = preparar_datos(DATA_DIR / INPUT_FILE, DATA_DIR / TARGET_FILE)
    x_train, y_train = datos["train"]
    x_val, y_val = datos["val"]
    x_test, y_test = datos["test"]

    dataset = GuitarAmpDataset(x_train, y_train, sequence_length=1024)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = NeuralAmp(hidden_size=64, num_layers=2).to(device)
    criterion = ESRLoss(pre_enfasis=PRE_ENFASIS)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    output_model_path = MODELS_DIR / MODEL_FILE
    mejor_val = float("inf")

    print("Iniciando el entrenamiento...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            # Solo cuentan las muestras posteriores al warm-up
            loss = criterion(outputs[:, WARMUP:, :], targets[:, WARMUP:, :])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(dataloader)
        val_esr = esr(y_val, predecir(model, x_val, device))

        # Guardar solo el mejor modelo según la validación
        marca = ""
        if val_esr < mejor_val:
            mejor_val = val_esr
            torch.save(model.state_dict(), output_model_path)
            marca = "  <- mejor modelo"
        print(f"Época [{epoch+1}/{EPOCHS}] - ESR train: {train_loss:.4f} - ESR validación: {val_esr:.4f}{marca}")

    # Evaluación final con el mejor modelo sobre el test (datos nunca vistos)
    model.load_state_dict(torch.load(output_model_path, map_location=device, weights_only=True))
    pred_test = predecir(model, x_test, device)
    print(f"\nESR en test (datos no vistos): {esr(y_test, pred_test):.4f}")

    # Guardar el fragmento de test (real y predicho) para escucharlo
    sf.write(str(DATA_DIR / "test_real_2.wav"), y_test, SAMPLE_RATE)
    sf.write(str(DATA_DIR / "test_IA_2.wav"), pred_test, SAMPLE_RATE)
    print(f"Modelo guardado en: {output_model_path}")
    print(f"Audios de test guardados en: {DATA_DIR}")


if __name__ == "__main__":
    try:
        train_model()
    except FileNotFoundError:
        print("\n[!] Error: No se encontraron los archivos .wav.")
        print(f"Verifica que los audios estén en: {DATA_DIR}")
