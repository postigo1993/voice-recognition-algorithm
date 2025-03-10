import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import speech_recognition as sr
import pyttsx3
from resemblyzer import VoiceEncoder, preprocess_wav


# -----------------------------
# Funciones y Clases Auxiliares
# -----------------------------

def speak (text):
    """Emite el texto en voz alta usando pyttsx3."""
    engine = pyttsx3.init ()
    engine.say (text)
    engine.runAndWait ()


def capture_audio (temp_filename="temp.wav", prompt=None):
    """
    Captura audio desde el micrófono y lo guarda en un archivo temporal.
    Si se especifica 'prompt', se lo dice al usuario antes de capturar.
    """
    if prompt:
        speak (prompt)
    r = sr.Recognizer ()
    with sr.Microphone () as source:
        audio = r.listen (source)
    with open (temp_filename, "wb") as f:
        f.write (audio.get_wav_data ())
    return temp_filename


def cosine_similarity (a, b):
    """Calcula la similitud coseno entre dos vectores."""
    return np.dot (a, b) / (np.linalg.norm (a) * np.linalg.norm (b))


# Dataset para identificación de hablantes
class SpeakerDataset (Dataset):
    """
    Dataset personalizado que asume la siguiente estructura:
      data/<nombre_hablante>/*.wav
    Extrae los embeddings de la voz usando el encoder de Resemblyzer.
    """

    def __init__ (self, data_dir, encoder):
        self.data = []
        self.labels = []
        self.speaker_names = []
        for speaker_dir in os.listdir (data_dir):
            speaker_path = os.path.join (data_dir, speaker_dir)
            if os.path.isdir (speaker_path):
                self.speaker_names.append (speaker_dir)
                for wav_file in glob.glob (os.path.join (speaker_path, "*.wav")):
                    self.data.append (wav_file)
                    self.labels.append (speaker_dir)
        # Mapear nombres de hablantes a índices numéricos
        self.speaker_to_idx = {speaker: idx for idx, speaker in enumerate (sorted (self.speaker_names))}
        self.labels = [self.speaker_to_idx[label] for label in self.labels]
        self.encoder = encoder

    def __len__ (self):
        return len (self.data)

    def __getitem__ (self, idx):
        wav_path = self.data[idx]
        label = self.labels[idx]
        # Preprocesar el audio y extraer el embedding
        wav = preprocess_wav (wav_path)
        embedding = self.encoder.embed_utterance (wav)
        return torch.tensor (embedding, dtype=torch.float32), torch.tensor (label, dtype=torch.long)


# Modelo sencillo de clasificación (MLP)
class SpeakerClassifier (nn.Module):
    def __init__ (self, input_dim, num_classes):
        super (SpeakerClassifier, self).__init__ ()
        self.fc1 = nn.Linear (input_dim, 128)
        self.relu = nn.ReLU ()
        self.fc2 = nn.Linear (128, num_classes)

    def forward (self, x):
        # x: (batch, input_dim)
        x = self.fc1 (x)
        x = self.relu (x)
        x = self.fc2 (x)
        return x


def train_classifier (model, dataloader, criterion, optimizer, num_epochs=20, device="cpu"):
    model.to (device)
    for epoch in range (num_epochs):
        model.train ()
        running_loss = 0.0
        for embeddings, labels in dataloader:
            embeddings = embeddings.to (device)
            labels = labels.to (device)
            optimizer.zero_grad ()
            outputs = model (embeddings)
            loss = criterion (outputs, labels)
            loss.backward ()
            optimizer.step ()
            running_loss += loss.item () * embeddings.size (0)
        epoch_loss = running_loss / len (dataloader.dataset)
        print (f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        # Guardar checkpoint del modelo
        checkpoint_path = f"modelo_epoch_{epoch + 1}.pth"
        torch.save (model.state_dict (), checkpoint_path)
        print (f"Checkpoint guardado en {checkpoint_path}")
    return model


def evaluate_classifier (model, dataloader, device="cpu"):
    model.eval ()
    correct = 0
    total = 0
    with torch.no_grad ():
        for embeddings, labels in dataloader:
            embeddings = embeddings.to (device)
            labels = labels.to (device)
            outputs = model (embeddings)
            _, preds = torch.max (outputs, 1)
            correct += (preds == labels).sum ().item ()
            total += labels.size (0)
    accuracy = correct / total
    print (f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def recognize_speaker (model, encoder, device="cpu"):
    """
    Captura audio desde el micrófono, extrae el embedding y clasifica el hablante.
    """
    temp_filename = "mic_input.wav"
    capture_audio (temp_filename, prompt="Habla ahora para identificar tu voz...")
    wav = preprocess_wav (temp_filename)
    embedding = encoder.embed_utterance (wav)
    os.remove (temp_filename)
    embedding_tensor = torch.tensor (embedding, dtype=torch.float32).unsqueeze (0).to (device)
    model.eval ()
    with torch.no_grad ():
        outputs = model (embedding_tensor)
        _, pred = torch.max (outputs, 1)
    return pred.item ()


# -----------------------------
# Función principal
# -----------------------------
def main ():
    # Inicializar el encoder de Resemblyzer
    encoder = VoiceEncoder ()

    # Directorio de datos: debe tener la estructura data/<nombre_hablante>/*.wav
    data_dir = "data"
    dataset = SpeakerDataset (data_dir, encoder)
    print ("Mapping de hablantes:", dataset.speaker_to_idx)
    dataloader = DataLoader (dataset, batch_size=16, shuffle=True)

    input_dim = 256  # Dimensión del embedding de Resemblyzer (por defecto 256)
    num_classes = len (dataset.speaker_to_idx)

    model = SpeakerClassifier (input_dim, num_classes)
    criterion = nn.CrossEntropyLoss ()
    optimizer = optim.Adam (model.parameters (), lr=0.001)
    device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")

    print ("Entrenando el clasificador de hablantes...")
    model = train_classifier (model, dataloader, criterion, optimizer, num_epochs=20, device=device)
    evaluate_classifier (model, dataloader, device=device)

    # Uso del micrófono para reconocer la voz de un hablante en tiempo real
    predicted_label = recognize_speaker (model, encoder, device=device)
    idx_to_speaker = {v: k for k, v in dataset.speaker_to_idx.items ()}
    speaker_name = idx_to_speaker.get (predicted_label, "Desconocido")
    print ("Voz reconocida como:", speaker_name)
    speak ("La voz reconocida es " + speaker_name)


if __name__ == "__main__":
    main ()









