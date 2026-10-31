import json
from json import JSONEncoder
from pathlib import Path

import sys

import torch

# Aquest script viu a realtime_cpp/ però fa servir el model i els pesos
# entrenats de la part de deep learning (../deep_learning).
BASE_DIR = Path(__file__).resolve().parent
DL_DIR = BASE_DIR.parent / "deep_learning"
sys.path.insert(0, str(DL_DIR))
from model import NeuralAmp  # noqa: E402

PTH_PATH = DL_DIR / "models" / "mi_ampli_entrenado.pth"
MODELS_DIR = BASE_DIR / "models"


class EncodeTensor(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super().default(obj)


def export_for_rtneural():
    device = torch.device("cpu")

    model = NeuralAmp(hidden_size=64, num_layers=2)
    model.load_state_dict(torch.load(
        PTH_PATH, map_location=device, weights_only=True))
    model.eval()

    output_path = MODELS_DIR / "mi_ampli_rtneural.json"
    with open(output_path, "w") as json_file:
        json.dump(model.state_dict(), json_file, cls=EncodeTensor)

    print(f"¡Éxito! Pesos para RTNeural guardados en: {output_path}")


if __name__ == "__main__":
    export_for_rtneural()
