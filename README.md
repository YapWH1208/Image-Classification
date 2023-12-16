# Image-Classification with Vision Transformer
> Original Paper: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

## Overview

This project explores the application of Transformer-based architectures for image classification tasks. Unlike traditional Convolutional Neural Networks (CNNs) commonly used in image classification, this project utilizes Transformer models which adapted to handle image data.

The goal is to investigate the effectiveness of Vision Transformer (ViT) models in image classification by leveraging self-attention mechanisms to capture global dependencies and relationships within images. This project aims to demonstrate the potential of Transformers in understanding and classifying visual content without relying on convolutional operations.

## Dataset

The structure of your dataset we accept is shown as follows:
```
data
├── Class_1
│   ├── img_1-1.jpg
│   ├── img_1-2.jpg
│   └── ...
├── Class_2
│   ├── img_2-1.jpg
│   ├── img_2-2.jpg
│   └── ...
├── Class_3
│   ├── img_3-1.jpg
│   ├── img_3-2.jpg
│   └── ...
└── ...
```

## Model Training

### Model Architecture

#### Vision Transformer (ViT)

The core architecture utilized in this project for image classification tasks is the Vision Transformer (ViT). Unlike traditional Convolutional Neural Networks (CNNs), the ViT model employs a transformer-based architecture, originally proposed for sequence-to-sequence tasks in natural language processing, adapted to handle image data.

##### Key Components:

1. **Patch Embeddings**: Images are divided into fixed-size patches, which are then linearly embedded to generate sequence inputs for the transformer.
2. **Positional Encodings**: Positional encodings are added to the patch embeddings to provide spatial information about the patches' locations within the image.
3. **Transformer Encoder**: The ViT model consists of multiple transformer encoder layers that process the sequence of patch embeddings using self-attention mechanisms to capture global dependencies.
4. **Classification Head**: A standard linear classification head is appended on top of the transformer encoder to predict the image's class label.

##### Hyperparameters
> Only for reference in case of 3 classes with 1000+ images in total

- **Batch Size**: 1280
- **Epochs**: 80
- **Learning Rate**: 0.001
- **Patch Size**: 16
- **Hidden Size**: 64
- **Number of Hidden Layers**: 2
- **Number of Attention Heads**: 3
- **Intermediate Size**: 256
- **Dropout Probability (Hidden)**: 0.04
- **Dropout Probability (Attention)**: 0.12

### Training Procedure

The training process involves using the provided dataset split into training and testing sets. The model is trained for 80 epochs using the AdamW optimizer with a learning rate of 0.001. Training progress is monitored via training and testing loss, alongside accuracy metrics.

## Usage

To replicate the experiment:

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/YapWH1208/Image-Classification.git

2. **Prepare Data**:

    Ensure the video dataset is placed in the appropriate directory (e.g., `/data`).

3. **Run Training**:

    ```bash
    python train.py
    ```

## Results

### Model Performance

The trained Vision Transformer (ViT) model exhibits compelling performance in image classification tasks, demonstrating its effectiveness in understanding and categorizing visual content. The evaluation metrics indicate the model's capability to accurately classify images into their respective classes.

#### Evaluation Metrics:

- **Accuracy**: Achieved an overall accuracy of 80.5% on the test dataset, showcasing the model's ability to correctly classify images.

## Contributors

<a href="https://github.com/YapWH1208">
  <img src="https://avatars.githubusercontent.com/u/107160166" alt="YapWH" width="50" height="50">
</a>
<a href="https://github.com/HawkingC">
  <img src="https://avatars.githubusercontent.com/u/107180053" alt="CaoRui" width="50" height="50">
</a>
<a href="https://github.com/ASTAR123">
  <img src="https://avatars.githubusercontent.com/u/119657996" alt="ASTAR123" width="50" height="50">
</a>
<a href="https://github.com/Lst0107">
  <img src="https://avatars.githubusercontent.com/u/108055070" alt="Lst0107" width="50" height="50">
</a>

## License

This project is licensed under the [Apache License 2.0](LICENSE).
