# Sakoe-Chiba Banded DTW Wake-Word Spotting

Sonon avoids heavy neural network speech models by utilizing **Banded Dynamic Time Warping (DTW)** with **DBA Barycenter Averaging**.

## Key Advantages

1. **Few-Shot Enrollment**: Requires only 1 to 2 voice samples to register a new command.
2. **Computational Complexity**: Prunes $O(N \cdot M)$ full-matrix search to $O(N \cdot R)$ along the Sakoe-Chiba diagonal corridor ($R \approx 10$).
3. **Throughput**: Processes $> 2,500,000$ audio samples/sec per thread.
4. **Offline & Edge Determinism**: Constant memory footprint with zero cloud latency.

Enroll your own voice directly in the [Playground Register Tab](/playground).
