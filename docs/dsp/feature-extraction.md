# Feature Extraction & Mel Filterbanks

Sonon extracts audio features in streaming 10 ms frames (160 samples @ 16 kHz):

- **Windowing**: Hann, Hamming, and Blackman windows.
- **Radix-2 Cooley-Tukey FFT**: 512-point spectral magnitude.
- **Mel Filterbank**: 13 to 40 triangular filter bins.
- **PCEN**: Per-Channel Energy Normalization replacing logarithmic compression for dynamic range resilience under high acoustic variance.
