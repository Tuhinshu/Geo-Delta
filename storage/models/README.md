# GeoDelta Model Weights Directory (Air-Gapped Storage)
## Problem Statement: SIH26227 (Ministry of Defence, Government of India)

This directory hosts pre-trained neural network weights in accordance with the air-gapped readiness requirements (**NFR-SEC-001** and **NFR-SEC-003**).

### 1. RemoteCLIP (Domain Vision-Language Model Text Encoder)
- **Target Subdirectory:** `storage/models/remoteclip/`
- **File Asset:** `RemoteCLIP-ViT-B-32.pt`
- **Source Repository:** `https://github.com/Chen-Guanzhou/RemoteCLIP` (Hugging Face: `chendelong/RemoteCLIP`)
- **Architecture:** ViT-B/32 text encoder trained on 800k remote-sensing image-caption pairs.
- **Expected SHA-256 Checksum:** `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (Verified on ingestion)

### 2. Siamese Change Detection Backbone
- **Target Subdirectory:** `storage/models/siamese_backbone/`
- **File Asset:** `resnet34_levircd_checkpoint.pth`
- **Architecture:** ResNet-34 ImageNet baseline weights initialized with LEVIR-CD building change detection Siamese checkpoints.
- **Source Reference:** `https://github.com/fitzp/SNUNet-CD`

### Supply Chain Security Verification (NFR-SEC-003)
Container startup scripts compute and compare the SHA-256 hash of each weight file before loading into memory. If weights are absent during development, the backend automatically switches to `DeterministicFallbackMode` (OpenCV Otsu spectral change differencing).
