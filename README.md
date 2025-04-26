# AI Car Editor

An AI-powered tool that automatically creates synchronized car edits by matching video clips to music beats.

## Overview

This project uses machine learning (XGBoost) to analyze car video clips and music tracks, then automatically generates a seamless edit that synchronizes visual transitions with audio beats. The tool can be trained on existing car edits to improve its cutting decisions. 

## Warning
This project was made for entertainment purposes only, this is not actively Maintained on!

## Features

- Automatic synchronization of video cuts with music beats
- Custom training capability to match your editing style
- Support for various video and audio formats
- Easy-to-use command line interface

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Its-BB/AI-Editor.git
   cd AI-Editor
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Directory Structure

```
AI-Editor/
│
├── main.py            # Main script for generating edits
├── trainer.py         # Script for training the model
│
├── Clips/             # Put your car video clips here
│
├── Songs/             # Put your music tracks here
│
├── Trainer/           # Put reference car edits here for training
│
└── models/            # Trained model files will be saved here
```

## Usage

### Generating an Edit

1. Place your car video clips in the `Clips` folder
2. Place your music track in the `Songs` folder
3. Run the main script:
   ```
   python main.py
   ```
4. Your generated edit will be saved in the project directory

### Training the Model

To train the model on your own editing style:

1. Place example car edit videos in the `Trainer` folder
2. Run the training script:
   ```
   python trainer.py
   ```
3. The trained model will be saved and used for future edits

## How It Works

1. **Audio Analysis**: Detects beats and energy levels in the music
2. **Video Analysis**: Processes clip content and motion 
3. **AI Prediction**: Uses machine learning to determine the best cut points
4. **Video Generation**: Combines clips with the music track based on AI predictions

## Requirements

See `requirements.txt` for the complete list of dependencies.

## License

This project is open source and available under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.