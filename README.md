# Human-Action-Estimation

## Overview

This project focuses on Human Action Estimation, an essential task in computer vision, where the goal is to recognize and classify human actions from video data. The primary objective is to create a model capable of accurately identifying three main classes of human actions: `gym`, `walk`, and `work`.

## Classes

- **gym**: Represents actions related to exercising or workouts in a gym environment.
- **walk**: Represents actions related to walking or strolling.
- **work**: Represents actions associated with working or performing tasks.

## Dataset

The project utilizes a custom dataset curated to encompass diverse scenarios and variations within the `gym`, `walk`, and `work` classes. Each class contains a collection of video sequences capturing various instances of the corresponding human action.

## Model Training

### Model Architecture

The Human Action Estimation model employed in this project is based on a Vision Transformer (ViT) architecture. The ViT model has been adapted and fine-tuned to process temporal sequences efficiently, leveraging self-attention mechanisms to capture spatiotemporal features crucial for accurate action recognition.

### Hyperparameters

- **Batch Size**: 128
- **Epochs**: 100
- **Learning Rate**: 0.001
- **Patch Size**: 16
- **Hidden Size**: 48
- **Number of Hidden Layers**: 4
- **Number of Attention Heads**: 4
- **Intermediate Size**: 192
- **Dropout Probability (Hidden)**: 0.0
- **Dropout Probability (Attention)**: 0.0

### Training Procedure

The training process involves using the provided dataset split into training and testing sets. The model is trained for 100 epochs using the AdamW optimizer with a learning rate of 0.001. Training progress is monitored via training and testing loss, alongside accuracy metrics.

## Usage

To replicate the experiment:

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/YapWH1208/Image-Classification.git

2. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare Data**:

    Ensure the video dataset is placed in the appropriate directory (e.g., `/data`).

4. **Run Training**:

    ```bash
    python train.py
    ```

## Results

The trained model achieves significant accuracy in recognizing `gym`, `walk`, and `work` actions, as observed during evaluation on the testing set. Detailed results and performance metrics can be found in the associated experiment logs.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to open a pull request or report any bugs or suggestions in the Issues section.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
