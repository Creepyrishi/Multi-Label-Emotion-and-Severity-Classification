# Multi-Label Emotion and Severity Classification from Video and Audio
This project trains a deep learning model to classify emotions and severity levels from audiovisual data using the RAVDESS dataset.
It uses 3D ResNet (r3d-18) for video frames and VGG16 for audio spectrograms, with fine-tuning on selected layers.

Features
- Extracts frames from video and spectrograms from audio.

- Fine-tunes pre-trained models on multi-label classification (emotion and severity).

- Designed for small batch processing and low-resource environments.

```
Dataset
RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song
```

## Model Architecture

- Video Branch: Fine-tuned r3d_18

- Audio Branch: Fine-tuned vgg16
