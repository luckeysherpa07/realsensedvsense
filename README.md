# Depth Warping: Spatial-Temporal Alignment of Event and Depth Sensors

This GitHub project implements a reusable pipeline for synchronizing and aligning data from an Intel RealSense D435i depth camera and a DVSense event camera.[file:1] Developed as part of a Master's thesis at Julius-Maximilians-Universität Würzburg, Institute of Computer Science, Winter Semester 2025/2026.[file:1] The pipeline addresses temporal mismatches and spatial misalignments between frame-based depth snapshots and asynchronous event streams.[file:1]

## Key Features
- **Temporal Alignment**: Builds signals from depth frames and event streams, then uses cross-correlation to find optimal time offset for synchronization.[file:1]
- **Spatial Alignment**: Performs camera calibration to estimate intrinsics and extrinsics, followed by 3D depth warping (unproject, transform, reproject) to map depth pixels to event camera coordinates.[file:1]
- **Visual Validation**: Overlays warped depth on event frames for verification.[file:1]
- **Demos**: Includes demonstrations for 3D depth, low-light (dark environments), motion blur (fast jittery movement), and high dynamic range (flashlight) scenarios.[file:1]

## Technical Overview
Event cameras output sparse events (x, y, p, t) with microsecond timestamps and high dynamic range (120 dB), complementing depth sensors like Intel RealSense D435i (stereo IR, up to 90 fps depth, 0.3-3m range).[file:1] Challenges include timestamp mismatches and differing fields of view/focal lengths due to side-by-side mounting.[file:1] Calibration uses checkerboard (10x7) for intrinsics/extrinsics; warping leverages rotation matrix R and translation t.[file:1]

## Usage
Run the pipeline for synchronization and warping on recorded data. Demos showcase fusion benefits in challenging conditions like low light, fast motion, and high DR.[file:1] Overlay warped depth on events to visualize alignment.[file:1]

## Challenges and Solutions
- IR projector patterns visible in events: Noted as encountered issue.[file:1]
- Calibration process: Detailed intrinsics estimation and stereo calibration inspired by NI Vision docs.[file:1]

## Future Work
Explore continuous-time models over cross-correlation, learning-based fusion, faster processing, and outdoor scenarios.[file:1]

## References
- OpenDIBR for depth warping concepts.[file:1]
- Various event-based vision papers (e.g., Tan et al. 2022 on spatio-temporal features).[file:1]


