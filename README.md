# voice-recognition-algorithm
This repository shows a voice recognition algorithm to train a model from scratch.

# Explanation

# Dataset and Embedding Extraction:
The dataset assumes that you have a folder named data in the same location as the script and subfolders named after each speaker (e.g. data/Alice, data/Bob, etc.). Each subfolder contains .wav files. For each file, a 256-dimensional embedding is extracted using Resemblyzer.
# Classification Model:
A simple MLP model (with two linear layers and a ReLU activation) is defined that takes the embedding as input and predicts the speaker.
# Training with Checkpoints:
The train_classifier function trains the model and, at the end of each epoch, saves a checkpoint to a file model_epoch_X.pth using torch.save(model.state_dict(), checkpoint_path). This allows the training progress to be saved.
# Real-Time Recognition:
The recognize_speaker function captures a voice sample from the microphone, extracts its embedding and uses the trained model to predict which speaker it corresponds to.
