add_executable(test-urwasm
    urwasm_api.cpp
    urwasm_main.cpp
)

target_link_libraries(test-urwasm PRIVATE llama Threads::Threads)

target_include_directories(test-urwasm PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)