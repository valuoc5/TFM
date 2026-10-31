#pragma warning(disable : 4996)
#include <RtAudio.h>
#include <RTNeural/RTNeural.h>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <windows.h>

#ifdef _DEBUG
#include <crtdbg.h>
#endif

namespace {

// Resuelve una ruta relativa al directorio del .exe (no al directorio de
// trabajo actual, que cambia segun desde donde se lance el programa).
std::filesystem::path exeRelativePath(const std::string& fileName) {
    char exePath[MAX_PATH];
    GetModuleFileNameA(nullptr, exePath, MAX_PATH);
    return std::filesystem::path(exePath).parent_path() / fileName;
}

// Arquitectura identica a la entrenada en model.py: LSTM(1->64) x2 capas + Linear(64->1).
// ModelT es un modelo de tamaño fijo en tiempo de compilacion: sin asignaciones
// de memoria en el camino de audio y con el hidden state persistente entre
// llamadas a forward() (cada llamada procesa UNA muestra y actualiza el estado).
using ModelType = RTNeural::ModelT<float, 1, 1,
    RTNeural::LSTMLayerT<float, 1, 64>,
    RTNeural::LSTMLayerT<float, 64, 64>,
    RTNeural::DenseT<float, 64, 1>>;

struct AmpContext {
    ModelType model;
};

bool loadModel(ModelType& model, const std::string& path) {
    std::ifstream jsonStream(path, std::ifstream::binary);
    if (!jsonStream.is_open()) {
        std::cerr << "No se pudo abrir el archivo de pesos: " << path << std::endl;
        return false;
    }

    nlohmann::json modelJson;
    jsonStream >> modelJson;

    auto& lstm_l0 = model.get<0>();
    RTNeural::torch_helpers::loadLSTM<float>(modelJson, "lstm.", lstm_l0, true, 0);
    auto& lstm_l1 = model.get<1>();
    RTNeural::torch_helpers::loadLSTM<float>(modelJson, "lstm.", lstm_l1, true, 1);
    auto& dense = model.get<2>();
    RTNeural::torch_helpers::loadDense<float>(modelJson, "linear.", dense);

    model.reset();
    return true;
}

// Callback de audio: procesa el bloque de entrada muestra a muestra a traves
// del modelo (el hidden state del LSTM se mantiene entre llamadas, no se
// reinicia en cada bloque).
int audioCallback(void* outputBuffer, void* inputBuffer, unsigned int nFrames,
                   double /*streamTime*/, RtAudioStreamStatus status, void* userData) {
    if (status) {
        std::cerr << "[Aviso] Underflow/overflow de audio detectado." << std::endl;
    }

    auto* ctx = static_cast<AmpContext*>(userData);
    const float* in = static_cast<const float*>(inputBuffer);
    float* out = static_cast<float*>(outputBuffer);

    for (unsigned int i = 0; i < nFrames; ++i) {
        float sample = in ? in[i] : 0.0f;
        out[i] = ctx->model.forward(&sample);
    }

    return 0;
}

void listDevices(RtAudio& audio, const std::vector<unsigned int>& deviceIds) {
    std::cout << "\nDispositivos de audio disponibles:" << std::endl;
    for (auto id : deviceIds) {
        RtAudio::DeviceInfo info = audio.getDeviceInfo(id);
        std::cout << "  [" << id << "] " << info.name
                   << "  (entradas: " << info.inputChannels
                   << ", salidas: " << info.outputChannels << ")"
                   << (info.isDefaultInput ? "  [ENTRADA POR DEFECTO]" : "")
                   << (info.isDefaultOutput ? "  [SALIDA POR DEFECTO]" : "")
                   << std::endl;
    }
    std::cout << std::endl;
}

unsigned int askDeviceId(const std::string& prompt, unsigned int defaultId) {
    std::cout << prompt << " [ENTER = " << defaultId << "]: ";
    std::string line;
    std::getline(std::cin, line);

    // Ignorar cualquier caracter que no sea digito (p.ej. un BOM UTF-8 o
    // espacios en blanco colados), en vez de asumir que la linea es un
    // numero limpio.
    std::string digitsOnly;
    for (char c : line) {
        if (std::isdigit(static_cast<unsigned char>(c))) {
            digitsOnly += c;
        }
    }

    if (digitsOnly.empty()) {
        return defaultId;
    }
    return static_cast<unsigned int>(std::stoul(digitsOnly));
}

} // namespace

int runApp() {
    std::cout << "=== Amplificador Virtual C++ (tiempo real, RTNeural) ===" << std::endl;

    AmpContext ctx;
    // Copiado junto al .exe por CMake (ver CMakeLists.txt). Se resuelve
    // relativo al .exe, no al directorio de trabajo actual, para que
    // funcione sin importar desde donde se lance el programa.
    std::string weights_path = exeRelativePath("mi_ampli_rtneural.json").string();
    if (!loadModel(ctx.model, weights_path)) {
        return -1;
    }
    std::cout << "Modelo cargado correctamente desde: " << weights_path << std::endl;

    RtAudio audio;
    std::cout << "API de audio en uso: " << audio.getApiDisplayName(audio.getCurrentApi()) << std::endl;
    auto deviceIds = audio.getDeviceIds();
    if (deviceIds.empty()) {
        std::cerr << "No se encontraron dispositivos de audio." << std::endl;
        return -1;
    }

    listDevices(audio, deviceIds);

    unsigned int inputId = askDeviceId("ID del dispositivo de ENTRADA (tu interfaz de guitarra)",
                                        audio.getDefaultInputDevice());
    unsigned int inputChannel = askDeviceId("Canal FISICO de entrada (1 = Input 1, 2 = Input 2, ...)", 1);
    unsigned int outputId = askDeviceId("ID del dispositivo de SALIDA (auriculares)",
                                         audio.getDefaultOutputDevice());
    unsigned int bufferFrames = askDeviceId("Tamaño de buffer en muestras (mas bajo = menos latencia, mas riesgo de xruns)", 128);

    RtAudio::StreamParameters inputParams;
    inputParams.deviceId = inputId;
    inputParams.nChannels = 1;
    inputParams.firstChannel = inputChannel - 1; // RtAudio es 0-indexado

    RtAudio::StreamParameters outputParams;
    outputParams.deviceId = outputId;
    outputParams.nChannels = 1;
    outputParams.firstChannel = 0;

    unsigned int sampleRate = 44100;

    RtAudio::StreamOptions streamOptions;
    streamOptions.flags = RTAUDIO_SCHEDULE_REALTIME;

    RtAudioErrorType err = audio.openStream(&outputParams, &inputParams, RTAUDIO_FLOAT32,
                                             sampleRate, &bufferFrames, &audioCallback,
                                             &ctx, &streamOptions);
    if (err != RTAUDIO_NO_ERROR) {
        std::cerr << "Error al abrir el stream de audio (codigo " << err << ")." << std::endl;
        return -1;
    }

    err = audio.startStream();
    if (err != RTAUDIO_NO_ERROR) {
        std::cerr << "Error al iniciar el stream de audio (codigo " << err << ")." << std::endl;
        return -1;
    }

    std::cout << "\nProcesando audio en tiempo real (buffer de " << bufferFrames
               << " muestras). Toca la guitarra." << std::endl;
    std::cout << "Presiona ENTER para detener...\n" << std::endl;
    std::cin.get();

    if (audio.isStreamRunning()) {
        audio.stopStream();
    }
    if (audio.isStreamOpen()) {
        audio.closeStream();
    }

    std::cout << "Detenido." << std::endl;
    return 0;
}

int main() {
#ifdef _DEBUG
    // Redirigir los diálogos de aserción de depuración a stderr en vez de un
    // popup bloqueante, para poder diagnosticar sin interrumpir al usuario.
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif
    try {
        return runApp();
    }
    catch (const std::exception& e) {
        std::cerr << "\n[EXCEPCION] " << e.what() << std::endl;
        return -2;
    }
    catch (...) {
        std::cerr << "\n[EXCEPCION DESCONOCIDA]" << std::endl;
        return -3;
    }
}
