# Transformer-Based Anomaly Detection for Acoustic Emission Signals

### **With Triton-Optimized Attention Kernels & GPU Profiling**

This project develops a **transformer encoder** for **acoustic emission (AE)** data from Directed Energy Deposition (DED) additive manufacturing and integrates **Triton-optimized GPU kernels** to accelerate transformer attention. The goal is to combine **domain-driven anomaly detection** with **high-performance ML systems**, demonstrating both modeling expertise and kernel-level optimization.

This project was executed and benchmarked on ASU’s **Sol GPU Cluster** (V100/A100).

---

# Overview

Traditional transformer models suffer performance bottlenecks in:

* **QKᵀ attention score computation**
* **Row-wise softmax**
* **Attention-value multiplication**

This project:

1. Builds a **baseline transformer** for AE anomaly detection
2. Profiles model latency using `torch.profiler`
3. Identifies attention as the dominant bottleneck
4. Implements a **custom Triton GPU kernel** to accelerate either

   * QKᵀ, or
   * row-wise softmax
5. Benchmarks PyTorch vs Triton versions
6. Integrates the optimized kernel into the transformer encoder

This is a **full systems+modeling pipeline** aimed at real performance gains.

---

# Key Features

### **AE → Transformer Modeling**

* Converts raw AE signals or spectrogram-derived features into sequences
* Builds a **PyTorch TransformerEncoder** for anomaly detection
* Supports classification, reconstruction loss, or embedding modeling

### **Attention Profiling**

Using `torch.profiler`:

* Breaks down GPU time per operator
* Compares scaling across sequence lengths (128 → 2048)
* Identifies attention kernels as the primary bottleneck

### **Triton Kernel Optimization**

Implements a custom Triton kernel for:

* **QKᵀ matmul** or
* **Row-wise softmax**

Optimizations include:

* Block tiling
* Tensor memory coalescing
* Vectorized loads
* In-register accumulation

Triton provides CUDA-level speed while writing Python-like code.

---

# Project Structure

```
ae-transformer-triton-opt/
│
├── models/
│   ├── transformer_baseline.py      # PyTorch baseline encoder
│   └── transformer_triton.py        # Encoder using Triton-accelerated ops
│
├── kernels/
│   ├── qk_matmul_triton.py          # Triton QKᵀ kernel
│   └── softmax_triton.py            # Triton softmax kernel
│
├── scripts/
│   ├── train_baseline.py            # Train model on AE data
│   ├── profile_transformer.py       # GPU profiling and trace logging
│   └── benchmark_kernels.py         # PyTorch vs Triton performance tests
│
├── README.md
└── requirements.txt                 # torch, pytorch-lightning, triton, numpy...
```

---

## 🔗 Related MATLAB AE Toolbox

AE preprocessing (from raw AE binaries to spectrograms) is handled by a separate MATLAB repository:

> **AE Data Handler: MATLAB Tools for Acoustic Emission Analysis**  
> GitHub: `Xinmiao-Luan/ae-data-handler`  
> This toolbox reads, merges, filters, and visualizes AE data from DED processes, including waveform plotting and spectrogram-style visualizations (e.g., `readMultipleFiles.m`, `selectData.m`, `plotWaveform.m`, `drawspectrum.m`).

This transformer project assumes you have already used that toolbox to export **spectrograms + labels** as `.mat` files.

---

# Methodology

### **1. AE Preprocessing**

* Use the AE Data Handler MATLAB toolbox to load and merge AE waveforms from DED (readMultipleFiles, selectData). Extract time windows and convert AE segments into spectrograms (S). For each segment, save a .mat file containing: S: spectrogram [n_freq, n_time] and label: anomaly indicator (0 = normal, 1 = anomaly). Create an index.csv listing each .mat file and label.
* In Python, the AEAESpectrogramDataset: Loads spectrograms from .mat, applies optional log-scaling + normalization and converts to transformer-ready sequences: [seq_len, input_dim] = [n_time, n_freq]. Pads/truncates to a fixed seq_len. This produces a clean, standardized input format for the models.
* AE sequences are normalized and padded to fixed lengths.

---

### **2. Baseline Transformer Encoder**

A compact model using:

* Linear projection → d_model
* Multi-head self-attention
* Feed-forward layers
* Positional encodings
* Dropout / layernorm

Trained using PyTorch Lightning with GPU acceleration.

---

### **3. Profiling the Transformer**

Example command:

```bash
python scripts/profile_transformer.py --seq_len 1024 --batch_size 32
```

Outputs:

* Chrome trace file
* Breakdown of ops sorted by time
* Attention kernels highlighted as bottlenecks

---

### **4. Triton Kernel Implementation**

Example Triton kernel:

```python
@triton.jit
def qk_matmul_kernel(Q, K, Out, ...):
    # BLOCK_M x BLOCK_N tiling
    # Load tiles of Q and K
    # Compute dot product in-register
    # Store partial results
```

Works similarly to CUDA but with **Python syntax**.

---

### **5. Benchmarking**

Run:

```bash
python scripts/benchmark_kernels.py
```

Plots:

* Latency vs sequence length
* PyTorch vs Triton speedup
* GPU utilization improvements

---

# Installation

```bash
conda create -n ae-triton python=3.10
conda activate ae-triton
pip install -r requirements.txt
```

---

# Quick Start

### Train baseline transformer:

```bash
python scripts/train_baseline.py
```

### Profile attention:

```bash
python scripts/profile_transformer.py
```

### Run Triton benchmarks:

```bash
python scripts/benchmark_kernels.py
```

---

# Contact & Extensions

Future extensions:

* Integrate FlashAttention-like fused kernels
* Add multi-head Triton kernels
* Explore AE-guided positional encodings
* Apply model to full DED build scans
