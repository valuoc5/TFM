"""Entrenamiento de un único modelo condicionado para varios amplificadores.

La red recibe en cada paso la muestra de guitarra limpia y un vector one-hot
que indica qué amplificador debe imitar (el "switch").
Reutiliza la carga, la alineación y la pérdida de train.py.
"""
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import NeuralAmp
from train import (cargar_canal, estimar_retardo, ESRLoss, esr, SAMPLE_RATE,
                   MARGEN_RETARDO, WARMUP, PRE_ENFASIS, MAX_GRAD_NORM,
                   TRAIN_FRAC, VAL_FRAC)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUT_DIR = BASE_DIR / "data" / "multi"

# ==========================================
# 0. CONFIGURACIÓN
# ==========================================
MEDIA_DIR = Path(r"C:\Program Files\ReaperProjects\instaidea\ii\Media")
CLEAN_FILE = "clean.wav"
AMPS = ["crunch", "dirty_coil", "acoustic", "high_gain"]  # el orden define el one-hot
MODEL_FILE = "multi_amp.pth"

EPOCHS = 200
PACIENCIA = 20          # parada temprana: épocas sin mejorar la validación
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SEQ_LEN = 1024


# ==========================================
# 1. DATOS
# ==========================================
def preparar_datos():
    """Devuelve, para cada amplificador, las partes train/val/test alineadas."""
    x_full = cargar_canal(MEDIA_DIR / CLEAN_FILE)
    x_full = x_full / np.max(np.abs(x_full))
    datos = {}
    for amp in AMPS:
        y = cargar_canal(MEDIA_DIR / f"{amp}.wav")
        n = min(len(x_full), len(y))
        x, y = x_full[:n], y[:n]

        lag_medido = estimar_retardo(x, y)
        lag = max(lag_medido - MARGEN_RETARDO, 0) if lag_medido > 0 else lag_medido
        if lag > 0:
            x, y = x[:-lag], y[lag:]
        elif lag < 0:
            x, y = x[-lag:], y[:lag]
        y = y / np.max(np.abs(y))

        n = len(x)
        i_train, i_val = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
        datos[amp] = {
            "train": (x[:i_train], y[:i_train]),
            "val": (x[i_train:i_val], y[i_train:i_val]),
            "test": (x[i_val:], y[i_val:]),
        }
        print(f"  {amp:12s} retardo medido {lag_medido:3d}, compensado {lag:3d} | "
              f"train {i_train / SAMPLE_RATE:.0f} s, val {(i_val - i_train) / SAMPLE_RATE:.0f} s, "
              f"test {(n - i_val) / SAMPLE_RATE:.0f} s")
    return datos


def con_condicion(x, k):
    """Añade a la señal (N,) las columnas one-hot del amplificador k -> (N, 1 + nº amps)."""
    cond = np.zeros((len(x), len(AMPS)), dtype=np.float32)
    cond[:, k] = 1.0
    return np.concatenate([x[:, None].astype(np.float32), cond], axis=1)


class MultiAmpDataset(Dataset):
    """Trozos de SEQ_LEN muestras de todos los amplificadores mezclados."""
    def __init__(self, datos):
        self.items = []  # (k, x, y, inicio)
        for k, amp in enumerate(AMPS):
            x, y = datos[amp]["train"]
            for i in range(len(x) // SEQ_LEN):
                self.items.append((k, x, y, i * SEQ_LEN))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        k, x, y, s = self.items[idx]
        x_in = torch.from_numpy(con_condicion(x[s:s + SEQ_LEN], k))
        y_out = torch.tensor(y[s:s + SEQ_LEN], dtype=torch.float32).unsqueeze(-1)
        return x_in, y_out


# ==========================================
# 2. EVALUACIÓN
# ==========================================
def predecir(model, x, k, device, chunk_size=44100):
    """Procesa un audio continuo para el amplificador k, manteniendo el estado."""
    model.eval()
    hidden, salida = None, []
    with torch.no_grad():
        for i in range(0, len(x), chunk_size):
            chunk = torch.from_numpy(con_condicion(x[i:i + chunk_size], k)).unsqueeze(0).to(device)
            lstm_out, hidden = model.lstm(chunk, hidden)
            salida.append(model.linear(lstm_out).view(-1).cpu().numpy())
    return np.concatenate(salida)


def evaluar(model, datos, parte, device):
    return {amp: esr(datos[amp][parte][1], predecir(model, datos[amp][parte][0], k, device))
            for k, amp in enumerate(AMPS)}


# ==========================================
# 3. ENTRENAMIENTO
# ==========================================
def train_multi():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en: {device}")
    print("Cargando audios...")
    datos = preparar_datos()

    dataset = MultiAmpDataset(datos)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Trozos de entrenamiento: {len(dataset)} ({len(dataloader)} lotes por época)")

    model = NeuralAmp(hidden_size=64, num_layers=2, input_size=1 + len(AMPS)).to(device)
    criterion = ESRLoss(pre_enfasis=PRE_ENFASIS)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    MODELS_DIR.mkdir(exist_ok=True)
    output_model_path = MODELS_DIR / MODEL_FILE
    mejor_val, sin_mejora = float("inf"), 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs[:, WARMUP:, :], targets[:, WARMUP:, :])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            running_loss += loss.item()

        val = evaluar(model, datos, "val", device)
        val_media = float(np.mean(list(val.values())))
        marca = ""
        if val_media < mejor_val:
            mejor_val, sin_mejora = val_media, 0
            torch.save(model.state_dict(), output_model_path)
            marca = "  <- mejor modelo"
        else:
            sin_mejora += 1
        detalle = " ".join(f"{a}={v:.3f}" for a, v in val.items())
        print(f"Época [{epoch+1}/{EPOCHS}] - ESR train: {running_loss / len(dataloader):.4f} - "
              f"ESR val media: {val_media:.4f} ({detalle}){marca}")
        if sin_mejora >= PACIENCIA:
            print(f"Parada temprana: {PACIENCIA} épocas sin mejorar.")
            break

    # Evaluación final con el mejor modelo
    model.load_state_dict(torch.load(output_model_path, map_location=device, weights_only=True))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\nESR en test (datos no vistos):")
    for k, amp in enumerate(AMPS):
        x_test, y_test = datos[amp]["test"]
        pred = predecir(model, x_test, k, device)
        print(f"  {amp:12s} {esr(y_test, pred):.4f}")
        sf.write(str(OUT_DIR / f"test_real_{amp}.wav"), y_test, SAMPLE_RATE)
        sf.write(str(OUT_DIR / f"test_IA_{amp}.wav"), pred, SAMPLE_RATE)
    print(f"Modelo guardado en: {output_model_path}")
    print(f"Audios de test en: {OUT_DIR}")


if __name__ == "__main__":
    train_multi()
