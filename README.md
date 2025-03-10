# Recognition and Identification System.

This repository contains a custom-built voice recognition system designed to identify speakers based on their unique vocal characteristics. The project uses the Resemblyzer library to extract 256-dimensional voice embeddings from audio files and trains a simple MLP classifier using PyTorch. The dataset is organized into subfolders (one per speaker) under a main data directory.

Key features include:

    Custom Training Pipeline: Extracts voice embeddings and trains a speaker classifier with checkpoints saved after each epoch.
    Real-Time Recognition: Captures live audio via a microphone, processes it to extract embeddings, and predicts the speaker's identity.
    Extensible Design: Easily adaptable to various applications such as voice-controlled access systems (e.g., electronic locks).

# Explanation

# Dataset and Embedding Extraction:
The dataset assumes that you have a folder named data in the same location as the script and subfolders named after each speaker (e.g. data/Alice, data/Bob, etc.). Each subfolder contains .wav files. For each file, a 256-dimensional embedding is extracted using Resemblyzer.
# Classification Model:
A simple MLP model (with two linear layers and a ReLU activation) is defined that takes the embedding as input and predicts the speaker.
# Training with Checkpoints:
The train_classifier function trains the model and, at the end of each epoch, saves a checkpoint to a file model_epoch_X.pth using torch.save(model.state_dict(), checkpoint_path). This allows the training progress to be saved.
# Real-Time Recognition:
The recognize_speaker function captures a voice sample from the microphone, extracts its embedding and uses the trained model to predict which speaker it corresponds to.

Dependencies include:

    Resemblyzer
    PyTorch
    SpeechRecognition
    pyttsx3
    NumPy

Feel free to explore, modify, and extend this project to suit your needs.
