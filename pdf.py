import fitz  # PyMuPDF
import os

def extract_and_save_text(pdf_path, chapters):
    # Проверка существования PDF файла
    if not os.path.isfile(pdf_path):
        print(f"Файл {pdf_path} не найден.")
        return

    document = fitz.open(pdf_path)
    total_pages = len(document)

    for i, (start, end) in enumerate(chapters):
        # Проверка корректности границ
        if start < 1 or end > total_pages or start > end:
            print(f"Некорректные границы для главы {i+1}: ({start}, {end})")
            continue

        text = ""
        for page_num in range(start - 1, end):  # Индексация страниц начинается с 0
            page = document.load_page(page_num)
            text += page.get_text()

        output_file_name = f"chapter_{i+1}.txt"
        with open(output_file_name, "w", encoding="utf-8") as output_file:
            output_file.write(text)
        print(f"Создан файл: {output_file_name}")

if __name__ == "__main__":
    pdf_path = "hard.pdf"

    # Укажите границы глав в формате [[начало_главы, конец_главы], [начало_главы, конец_главы], ...]
    chapters = [
        [2, 3],   # Charles Babbage’s Analytical Engine
        [4, 4],   # ENIAC
        [5, 5],   # IBM PC
        [6, 7],   # The Intel 8088 microprocessor
        [8, 8],   # The Intel 80286 and 80386 microprocessors
        [9, 10],  # The iPhone
        [11, 13], # Moore’s law
        [14, 17], # Computer architecture
        [18, 21], # The 6502 microprocessor
        [22, 24],  # The 6502 instruction set
        [27, 52],   # Chapter 2: Digital Logic
        [28, 28],  # Technical requirements
        [28, 29],  # Electrical circuits
        [29, 30],  # The transistor
        [30, 34],  # Logic gates
        [35, 37],  # Latches
        [38, 39],  # Flip-flops
        [40, 40],  # Registers
        [41, 42],  # Adders
        [43, 43],  # Propagation delay
        [44, 44],  # Clocking
        [45, 45],  # Sequential logic
        [46, 46],  # Hardware description languages
        [47, 50],   # VHDL
        [54, 54],  # Technical requirements
        [54, 55],  # A simple processor
        [55, 56],  # Control unit
        [57, 57],  # Executing an instruction – a simple example
        [58, 63],  # Arithmetic logic unit
        [64, 64],  # Registers
        [65, 65],  # The instruction set
        [66, 66],  # Addressing modes
        [66, 66],  # Immediate addressing mode
        [67, 67],  # Absolute addressing mode
        [68, 69],  # Absolute indexed addressing mode
        [70, 70],  # Indirect indexed addressing mode
        [71, 71],  # Instruction categories
        [72, 72],  # Memory load and store instructions
        [72, 72],  # Register-to-register data transfer instructions
        [72, 72],  # Stack instructions
        [73, 73],  # Arithmetic instructions
        [74, 74],  # Logical instructions
        [74, 75],  # Branching instructions
        [75, 75],  # Subroutine call and return instructions
        [75, 75],  # Processor flag instructions
        [75, 75],  # Interrupt-related instructions
        [76, 76],  # No operation instruction
        [76, 79],  # Interrupt processing
        [80, 81],  # Input/output operations
        [82, 82],  # Programmed I/O
        [82, 83],  # Interrupt-driven I/O
        [83, 83],   # Direct memory access
        [88, 88],  # Technical requirements
        [88, 89],  # Memory subsystem
        [89, 91],  # Introducing the MOSFET
        [92, 93],  # Constructing DRAM circuits with MOSFETs
        [92, 93],  # The capacitor
        [94, 95],  # The DRAM bit cell
        [96, 98],  # DDR5 SDRAM
        [99, 99],  # Graphics DDR
        [99, 99],  # Prefetching
        [100, 101],# I/O subsystem
        [100, 101],# Parallel and serial data buses
        [102, 103],# PCI Express
        [104, 104],# SATA
        [105, 105],# M.2
        [105, 105],# USB
        [106, 106],# Thunderbolt
        [106, 107],# Graphics displays
        [108, 108],# VGA
        [108, 108],# DVI
        [109, 109],# HDMI
        [109, 109],# DisplayPort
        [110, 110],# Network interface
        [110, 111],# Ethernet
        [111, 112],# Wi-Fi
        [112, 112],# Keyboard and mouse
        [112, 113],# Keyboard
        [113, 113],# Mouse
        [114, 115], # Modern computer system specifications
        [118, 118],# Technical requirements
        [118, 120],# Device drivers
        [119, 120],# The parallel port
        [121, 122],# PCIe device drivers
        [122, 123],# Device driver structure
        [124, 125],# BIOS
        [126, 126],# UEFI
        [127, 129],# The boot process
        [128, 129],# BIOS boot
        [128, 129],# UEFI boot
        [130, 130],# Trusted boot
        [131, 132],# Embedded devices
        [131, 141],# Operating systems
        [133, 135],# Processes and threads
        [136, 140],# Scheduling algorithms and process priority
        [141, 141], # Multiprocessing
        [146, 146],# Technical requirements
        [146, 151],# Real-time computing
        [148, 151],# Real-time operating systems
        [152, 161],# Digital signal processing
        [152, 154],# ADCs and DACs
        [155, 156],# DSP hardware features
        [157, 158],# Signal processing algorithms
        [157, 158],# Convolution
        [158, 159],# Digital filtering
        [159, 161],# Fast Fourier transform (FFT)
        [162, 163],# GPU processing
        [164, 164],# GPUs as data processors
        [164, 164],# Big data
        [165, 166],# Deep learning
        [167, 168], # Examples of specialized architectures
        [173, 173],# Technical requirements
        [174, 177],# The von Neumann, Harvard, and modified Harvard architectures
        [174, 175],# The von Neumann architecture
        [176, 176],# The Harvard architecture
        [177, 177],# The modified Harvard architecture
        [178, 191],# Physical and virtual memory
        [182, 185],# Paged virtual memory
        [186, 186],# Page status bits
        [187, 188],# Memory pools
        [189, 191], # Memory management unit
        [196, 196],# Technical requirements
        [196, 208],# Cache memory
        [198, 199],# Multilevel processor caches
        [199, 199],# Static RAM
        [200, 200],# Level 1 cache
        [201, 204],# Direct-mapped cache
        [205, 206],# Set associative cache
        [207, 208],# Processor cache write policies
        [209, 210],# Level 2 and level 3 processor caches
        [211, 219],# Instruction pipelining
        [214, 214],# Superpipelining
        [215, 216],# Pipeline hazards
        [217, 217],# Micro-operations and register renaming
        [218, 218],# Conditional branches
        [220, 220],# Simultaneous multithreading
        [221, 221], # SIMD processing
        [225, 225],# Technical requirements
        [226, 229],# Privileged processor modes
        [226, 229],# Handling interrupts and exceptions
        [230, 231],# Protection rings
        [232, 232],# Supervisor mode and user mode
        [233, 233],# System calls
        [234, 239],# Floating-point arithmetic
        [237, 238],# The 8087 floating-point coprocessor
        [239, 239],# The IEEE 754 floating-point standard
        [240, 242],# Power management
        [241, 242],# Dynamic voltage frequency scaling
        [242, 243],# Hibernation and sleep modes
        [244, 246],# Trusted computing
        [247, 247],# Trusted platform modules
        [248, 248], # Intel Software Guard Extensions (SGX)
        [250, 250],# Technical requirements
        [250, 255],# The x86 architecture
        [250, 250],# The Intel Pentium
        [251, 251],# Out-of-order execution
        [252, 253],# The Intel Core architecture
        [253, 253],# AVX and AVX2 instructions
        [254, 255],# AVX-512 instructions
        [256, 258],# The Arm architecture
        [257, 257],# The 32-bit Arm architecture
        [258, 258],# The 64-bit Arm architecture
        [259, 267],# Vector processing
        [260, 260],# Vector arithmetic
        [261, 262],# Scalar arithmetic
        [263, 263],# The Intel MMX instruction set
        [263, 264],# SIMD instructions
        [264, 265],# Single instruction multiple threads (SIMT)
        [266, 267],# Vector instruction challenges
        [268, 276],# Instruction sets for neural network processing
        [270, 271],# NVIDIA GPU Tensor Cores
        [272, 273],# Google TPU
        [274, 276],# CPU deep learning extensions
        [277, 287],# Reconfigurable processing
        [278, 278],# Field-programmable gate arrays
        [279, 279],# FPGA logic blocks
        [280, 281],# FPGA interconnects
        [282, 283],# Partial reconfiguration
        [284, 286],# FPGA applications
        [287, 287], # Emulation with FPGAs
        [294, 294],# Technical requirements
        [295, 298],# Overview of the RISC-V architecture
        [297, 298],# Five RISC-V base integer instruction sets
        [299, 300],# Memory addressing in RISC-V
        [301, 302],# The compressed instruction set
        [303, 304],# Multiplication and division instructions
        [305, 307],# Bitwise logical instructions
        [308, 310],# The floating-point instruction set
        [311, 313],# Atomic instructions
        [314, 317], # Optional vector instructions
        [320, 320],# Technical requirements
        [320, 325],# Virtual machines and hypervisors
        [322, 325],# The IBM VM operating system
        [326, 329],# Nested page tables
        [327, 327],# Hardware support for virtualization
        [328, 329],# Virtualization of I/O operations
        [330, 335],# Type 1 and Type 2 hypervisors
        [332, 333],# Xen
        [334, 335],# VMware
        [336, 337],# KVM
        [338, 339], # Containerization
        [342, 342],# Technical requirements
        [342, 346],# ASICs
        [345, 346],# Characteristics of ASICs
        [347, 348],# ASIP architectures
        [349, 351],# Machine learning processors
        [352, 353],# Google's TPU
        [354, 355],# Systolic arrays
        [356, 359],# Data flow architectures
        [358, 359],# Data flow graphs
        [360, 363],# Neuromorphic processors
        [364, 365], # Examples of domain-specific architectures
        [370, 370],# Technical requirements
        [370, 373],# Hardware roots of trust
        [372, 373],# Trusted platform modules
        [374, 377],# Confidential computing
        [376, 377],# Secure enclaves
        [378, 380],# Security information and event management (SIEM)
        [379, 380],# The zero trust model
        [381, 382],# Secure boot
        [383, 384],# Secure element chips
        [385, 388],# Hardware security modules (HSMs)
        [389, 391],# Encrypted memory
        [392, 393], # Case studies of confidential computing architectures
        [396, 396],# Technical requirements
        [396, 400],# Blockchain architecture
        [398, 400],# Distributed ledger technology
        [401, 405],# Blockchain consensus algorithms
        [403, 405],# Proof of work
        [406, 409],# Bitcoin mining hardware
        [407, 409],# ASIC miners
        [410, 414],# Ethereum mining hardware
        [412, 414],# GPU miners
        [415, 418], # Mining farms and energy consumption
        [421, 446], # Chapter 16: Self-Driving Vehicle Architectures
        [422, 422],# Technical requirements
        [422, 423],# Self-driving car subsystems
        [423, 426],# Autonomous vehicle perception
        [425, 426],# LIDAR
        [427, 429],# Radar
        [429, 430],# Ultrasonic sensors
        [430, 430],# Visual cameras
        [431, 435],# Autonomous vehicle decision making
        [432, 433],# Path planning
        [434, 435],# Obstacle avoidance
        [436, 440],# Autonomous vehicle control
        [438, 439],# Steering control
        [440, 440],# Throttle and brake control
        [441, 445], # Examples of self-driving vehicle platforms
        [450, 450],# Technical requirements
        [451, 456],# Quantum computing principles
        [452, 453],# Qubits
        [454, 454],# Superposition
        [455, 456],# Entanglement
        [457, 459],# Quantum gates and circuits
        [458, 459],# Quantum gates
        [460, 462],# Quantum algorithms
        [463, 465],# Quantum hardware platforms
        [466, 470],# Other emerging architectures
]


    extract_and_save_text(pdf_path, chapters)

