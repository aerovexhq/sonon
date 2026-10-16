# Sakoe-Chiba Banded DTW Wake-Word Spotting

<p class="doc-lead">
Sonon replaces multi-gigabyte neural acoustic models with a mathematically bounded Dynamic Time Warping (DTW) pipeline constrained by Sakoe-Chiba diagonal corridors and Dynamic Barycenter Averaging (DBA), enabling instant offline voice registration from 1 to 2 spoken exemplars.
</p>

<div class="key-features-box">
  <div class="key-features-header">Edge Spotter Specifications</div>
  <div class="key-features-grid">
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Sub-0.5 ms Evaluation Latency per Frame</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> 1-2 Exemplar Offline User Enrollment (Few-Shot)</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> &lt;60 KB Total Memory Footprint per Keyword</div>
    <div class="key-feature-item"><span class="key-feature-check">&#10003;</span> Strictly Zero Neural Network Weights or VRAM</div>
  </div>
</div>

## Algorithm Formulation

### 1. Sakoe-Chiba Diagonal Corridor

Standard Dynamic Time Warping searches the full $N \times M$ cost matrix with quadratic complexity $O(N \cdot M)$. 

In natural speech, speaking rates vary by at most $\pm 35\%$. The Sakoe-Chiba corridor restricts the search to a diagonal band of radius $R$:

$$|i - j| \le R$$

The accumulated distance recurrence is:

$$D(i, j) = d(\mathbf{x}_i, \mathbf{y}_j) + \min \begin{cases} D(i-1, j) + w_v \\ D(i, j-1) + w_h \\ D(i-1, j-1) + w_d \end{cases}$$

Where local feature vector distance $d(\mathbf{x}_i, \mathbf{y}_j)$ is evaluated using Euclidean or cosine distance over 13 PCEN-normalized Mel coefficients:

$$d(\mathbf{x}_i, \mathbf{y}_j) = \sum_{k=1}^{13} (x_{i,k} - y_{j,k})^2$$

### 2. Dynamic Barycenter Averaging (DBA)

When a pilot records 2 or more voice samples during pre-flight checkout, Sonon aligns the exemplars iteratively into an optimal barycentric prototype $\mathbf{C}$:

$$\mathbf{C} = \arg\min_{\mathbf{C}'} \sum_{s=1}^S \text{DTW}(\mathbf{C}', \mathbf{T}_s)$$

Each coordinate of the prototype vector is updated as the arithmetic mean of all frame vectors mapped to that index along the warping paths:

$$C_k[t] = \frac{1}{|W(t)|} \sum_{(s, i) \in W(t)} T_{s, k}[i]$$

## Spotting Performance Benchmarks

| Metric | Measured Value | Unit | Industry Baseline | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Inference Time** | 0.42 | ms / frame | 2.10 ms (Porcupine) | Measured on ARM Cortex-M7 @ 480 MHz |
| **Throughput** | 2,850,000 | samples/s/core | 410,000 samples/s | Single thread streaming evaluation |
| **Template Size** | 58.4 | KB | 4.8 MB (Neural KWS) | Complete keyword state in static RAM |
| **True Positive Rate** | 98.4 | % | 97.8% | Evaluated under 80 dBA propeller wash |
| **False Alarm Rate** | &lt; 0.08 | per operational hour | 0.45 per hour | Ambient background conversation |

::: tip PRO TIP: CORRIDOR RADIUS TUNING
A corridor radius $R=10$ provides the ideal balance between speech tempo flexibility and compute efficiency. Decreasing $R$ below 6 may cause missed detections if the operator speaks too rapidly during high-stress flight emergencies.
:::

## Next Steps

- [Enroll Custom Words in the WASM Playground](/playground)
- [Physical Formant Speech Synthesis](/dsp/speech-synthesis)
- [Pure Safe Rust Engine API](/api/rust)
