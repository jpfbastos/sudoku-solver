# Sudoku Solver and CNN-Based Digit Recognition

This project is a Sudoku solver which combines two main components: OpenCV for image processing, and a Convolutional Neural Network (CNN) for digit recognition. The CNN can be trained using either the MNIST or Typographical MNIST (TMNIST) datasets. The Sudoku solver takes an image of a Sudoku puzzle (saved or using the camera), extracts the digits and solves the puzzle.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
  - [Sudoku Solver](#sudoku-solver-explanation)
  - [CNN Model](#cnn-model-explanation)
- [Examples](#examples)

## Features
  - Image preprocessing with OpenCV.
  - Pretrained model loading to avoid retraining.
  - Digit extraction using Tensorflow Keras CNN for digit recognition.
  - Sudoku solving via backtracking algorithm.
  - Overlay of the solution on the original image.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/jpfbastos/sudoku-solver.git
    ```
2. Navigate to the repository directory
    ```
    cd sudoku-solver
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Solver**: Use the following command to solve the Sudoku puzzle (leave --input blank if camera is being used):
    ```bash
    python main.py --input path/to/your/sudoku_image.png --database (tmnist or mnist) 
    ```
2. **Output**: The program will display the original image with the solved Sudoku grid overlaid.

### Debug

Extra intermediate steps can be shown using the debug option
  ```bash
  python main.py --input path/to/your/sudoku_image.png --database (tmnist or mnist) --debug (read_puzzle, extract_digits or all)
  ```

## How It Works

### Sudoku Solver Explanation

1. **Image Preprocessing**:
    - Convert the image to grayscale and blur to reduce noise
    - Use thresholding to convert to a binary image
    - Detect contours to identify the Sudoku grid
    - Extract the grid

2. **Digit Recognition**:
    - Use `cv2.findContours` to locate digits within each cell
    - Apply `cv2.boundingRect` to create bounding boxes around detected digits
    - Use the trained CNN model to predict each digit

3. **Sudoku Solving**:
    - Implement a backtracking algorithm to solve the puzzle
    - Recursively fill in the grid by checking for valid numbers in each cell

4. **Visualization**:
    - Overlay the solved numbers onto the original grid using `cv2.putText`
    - Display the resulting image

### CNN Model Explanation

- Two Conv2D layers for feature extraction.
- MaxPooling2D layers for downsampling.
- Dropout layer for regularisation.
- Flatten layer to convert 2D feature maps into 1D.
- Dense layers for classification.

- Load the saved model and weights.
- Predict the class of an input digit image.

## Examples

Here are some examples of how the project works:

**Input Image**:
![Input Image](examples/sudoku_example.png)

**Output Image**:
![Output Image](output/sudoku_solution.png)