set(CMAKE_SYSTEM_NAME WASI)
set(CMAKE_SYSTEM_PROCESSOR wasm32)

set(CMAKE_C_COMPILER   "$ENV{WASI_SDK}/bin/clang")
set(CMAKE_CXX_COMPILER "$ENV{WASI_SDK}/bin/clang++")

set(CMAKE_SYSROOT "$ENV{WASI_SDK}/share/wasi-sysroot")

set(CMAKE_C_FLAGS   "-O3")
set(CMAKE_CXX_FLAGS "-O3 -fno-rtti -fno-exceptions")
