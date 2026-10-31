cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(TFM LANGUAGES CXX C)

# Configuración estándar de C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# RtAudio: I/O de audio en tiempo real (captura guitarra / reproduce salida)
set(RTAUDIO_BUILD_TESTING OFF CACHE BOOL "" FORCE)
# ASIO da latencias mucho menores que WASAPI en Windows; se compila junto a
# WASAPI, que queda como fallback automático si no hay driver ASIO instalado
# (RtAudio prueba las APIs compiladas en ese orden en el constructor por defecto).
set(RTAUDIO_API_ASIO ON CACHE BOOL "" FORCE)
FetchContent_Declare(
  rtaudio
  GIT_REPOSITORY https://github.com/thestk/rtaudio.git
  GIT_TAG        6.0.1
)
FetchContent_MakeAvailable(rtaudio)

# RTNeural: motor de inferencia liviano para tiempo real (sin LibTorch en el
# .exe final; LibTorch sigue usandose solo en Python para entrenar/exportar).
FetchContent_Declare(
  rtneural
  GIT_REPOSITORY https://github.com/jatinchowdhury18/RTNeural.git
  GIT_TAG        1fb1f075a5d66e85bfc8f488c3f3626840cb3a1d
)
FetchContent_MakeAvailable(rtneural)

# Crear el ejecutable apuntando a tu main.cpp
add_executable(NeuralAmpCpp main.cpp)

# Enlazar RtAudio (dinamico) y RTNeural (estatico, header-only)
target_link_libraries(NeuralAmpCpp PRIVATE rtaudio RTNeural)

# Eigen limita por defecto los objetos fijos en stack a ~128KB; nuestras
# matrices LSTM combinadas (256x129 floats) lo superan por poco. El stack de
# Windows es 1MB por defecto, hay margen de sobra.
target_compile_definitions(NeuralAmpCpp PRIVATE EIGEN_STACK_ALLOCATION_LIMIT=0)

# Configurar el estándar en el target concreto
set_property(TARGET NeuralAmpCpp PROPERTY CXX_STANDARD 17)

# Copiar la DLL de RtAudio junto al .exe para poder ejecutarlo sin tocar el PATH
if(MSVC)
  add_custom_command(TARGET NeuralAmpCpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:rtaudio>
    $<TARGET_FILE_DIR:NeuralAmpCpp>)
endif()

# Copiar los pesos entrenados junto al .exe (main.cpp los carga con una ruta
# relativa: "mi_ampli_rtneural.json") para no depender de una ruta absoluta.
add_custom_command(TARGET NeuralAmpCpp POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${CMAKE_SOURCE_DIR}/models/mi_ampli_rtneural.json
  $<TARGET_FILE_DIR:NeuralAmpCpp>)
