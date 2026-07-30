# Generated Cmake Pico project file

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Initialise pico_sdk from installed location
# (note this can come from environment, CMake cache etc)

# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.3.0)
set(toolchainVersion 15_2_Rel1)
set(picotoolVersion 2.3.0)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
set(PICO_BOARD pico2_w CACHE STRING "Board type")

# Pull in Raspberry Pi Pico SDK (must be before project)
include(pico_sdk_import.cmake)

project(lm-pico2w C CXX ASM)

# Initialise the Raspberry Pi Pico SDK
pico_sdk_init()

# ---------------------------------------------------------------------------
# llama2-pico: llama2.c inference, weights embedded in flash, KV cache in PSRAM
#
# Drop stories260K.bin and tok512.bin (from karpathy/llama2.c) into this same
# directory before building. They get objcopy'd straight into .rodata as
# linker symbols - no filesystem, no copying to RAM.
# ---------------------------------------------------------------------------

set(LLAMA_MODEL_BIN ${CMAKE_CURRENT_LIST_DIR}/stories260K.bin)
set(LLAMA_TOK_BIN   ${CMAKE_CURRENT_LIST_DIR}/tok512.bin)

if(NOT EXISTS ${LLAMA_MODEL_BIN})
    message(WARNING "stories260K.bin not found in ${CMAKE_CURRENT_LIST_DIR} - copy it there before building")
endif()
if(NOT EXISTS ${LLAMA_TOK_BIN})
    message(WARNING "tok512.bin not found in ${CMAKE_CURRENT_LIST_DIR} - copy it there before building")
endif()

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/stories260K_bin.o
    COMMAND ${CMAKE_OBJCOPY} -I binary -O elf32-littlearm -B arm
            --rename-section .data=.rodata,alloc,load,readonly,data,contents
            stories260K.bin ${CMAKE_CURRENT_BINARY_DIR}/stories260K_bin.o
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    DEPENDS ${LLAMA_MODEL_BIN}
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tok512_bin.o
    COMMAND ${CMAKE_OBJCOPY} -I binary -O elf32-littlearm -B arm
            --rename-section .data=.rodata,alloc,load,readonly,data,contents
            tok512.bin ${CMAKE_CURRENT_BINARY_DIR}/tok512_bin.o
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    DEPENDS ${LLAMA_TOK_BIN}
    VERBATIM
)

add_custom_target(llama_blobs DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/stories260K_bin.o
    ${CMAKE_CURRENT_BINARY_DIR}/tok512_bin.o
)

add_executable(llama2-pico
    llama_main.c
    model.c
    tokenizer.c
    psram_spi.c
)
add_dependencies(llama2-pico llama_blobs)

# Link the objcopy'd blobs in directly - this is how the model/tokenizer symbols appear
target_link_options(llama2-pico PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/stories260K_bin.o
    ${CMAKE_CURRENT_BINARY_DIR}/tok512_bin.o
)
set_target_properties(llama2-pico PROPERTIES LINK_DEPENDS
    "${CMAKE_CURRENT_BINARY_DIR}/stories260K_bin.o;${CMAKE_CURRENT_BINARY_DIR}/tok512_bin.o"
)

pico_set_program_name(llama2-pico "llama2-pico")
pico_set_program_version(llama2-pico "0.1")

pico_generate_pio_header(llama2-pico ${CMAKE_CURRENT_LIST_DIR}/psram_spi.pio)

target_compile_definitions(llama2-pico PRIVATE
    PSRAM_PIN_CS=16
    PSRAM_PIN_SCK=17
    PSRAM_PIN_MOSI=18
    PSRAM_PIN_MISO=19
)

pico_enable_stdio_uart(llama2-pico 0)
pico_enable_stdio_usb(llama2-pico 1)

target_include_directories(llama2-pico PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(llama2-pico
    pico_stdlib
    hardware_spi
    hardware_i2c
    hardware_dma
    hardware_pio
    hardware_interp
    hardware_timer
    hardware_watchdog
    hardware_clocks
)

pico_add_extra_outputs(llama2-pico)