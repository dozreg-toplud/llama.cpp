# examples/urwasm/CMakeLists.txt

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "wasm32")

    # ========================
    # WASM BUILD
    # ========================

    add_executable(urwasm urwasm_api.cpp)

    target_link_libraries(urwasm PRIVATE llama)

    target_include_directories(urwasm PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    target_link_options(urwasm PRIVATE
        -Wl,--no-entry
        -Wl,--export=llama_cpp_init_from_gguf_bytes
        -Wl,--export=infer_c_string
        -Wl,--export=malloc
        -Wl,--export=free
        -Wl,--export-memory
    )

else()

    # ========================
    # NATIVE BUILD
    # ========================

    add_executable(test-urwasm
        urwasm_api.cpp
        urwasm_main.cpp
    )

    target_link_libraries(test-urwasm PRIVATE llama)

    target_include_directories(test-urwasm PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

endif()