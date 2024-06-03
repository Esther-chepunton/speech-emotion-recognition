#libraries
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib

#Extraction of data
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract MFCCs (texture of the audio)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)

    # Extract Chroma (pitch)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_processed = np.mean(chroma.T, axis=0)

    # Extract Mel Spectrogram (spectral properties of audio)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_processed = np.mean(mel.T, axis=0)

    # Combine features into a single array
    features = np.hstack([mfccs_processed, chroma_processed, mel_processed])
    
    return features

# Function to load data and labels from the TESS dataset
def load_data_and_labels(data_dir):
    file_paths = []
    labels = []
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad', 'surprise']
    
    for emotion in emotions:
        emotion_dir = os.path.join(data_dir, f'OAF_{emotion}')
        if os.path.isdir(emotion_dir):
            for file_name in os.listdir(emotion_dir):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(emotion_dir, file_name)
                    file_paths.append(file_path)
                    labels.append(emotion)
                    print(f"Loaded file: {file_path} with label: {emotion}")  # Debugging statement
    return file_paths, labels

# Main script
if __name__ == "__main__":
    # specify the path to the dataset
    data_dir = r"C:\Users\Esther\Desktop\speech-emotion\archive (2)\TESS Toronto emotional speech set data"  # Update with the path to your TESS dataset
    
    # Load data and labels
    file_paths, labels = load_data_and_labels(data_dir)
    
    if not file_paths:
        print("No files found. Please check your data directory and structure.")
    else:
        # Extract features for all files
        data = []
        for file_path in file_paths:
            try:
                features = extract_features(file_path)
                data.append(features)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        
        # Convert to numpy arrays
        data = np.array(data)
        labels = np.array(labels)
        
        if data.size == 0:
            print("No data to train on. Ensure the audio files are correctly processed.")
        else:
            # Encode labels
            label_encoder = LabelEncoder()
            labels_encoded = label_encoder.fit_transform(labels)
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)
            
            # Train SVM classifier
            clf = SVC(kernel='linear', random_state=42)
            clf.fit(X_train, y_train)
            
            # Evaluate classifier on testing data
            accuracy = clf.score(X_test, y_test)
            print("Accuracy: {:.2f}".format(accuracy))
            
            # Save the trained model
            model_path = 'trained_emotion_model.pkl'
            joblib.dump(clf, model_path)
            print(f"Model saved to {model_path}")
            
            # Save the label encoder
            label_encoder_path = 'label_encoder.pkl'
            joblib.dump(label_encoder, label_encoder_path)
            print(f"Label encoder saved to {label_encoder_path}")

            # Load the trained model and label encoder
            clf = joblib.load(model_path)
            label_encoder = joblib.load(label_encoder_path)
            
            # Predict emotions for a new audio file
            example_file_path = r"path_to_new_audio_file.wav"  
            new_features = extract_features(example_file_path)
            predicted_emotion_encoded = clf.predict([new_features])
            predicted_emotion = label_encoder.inverse_transform(predicted_emotion_encoded)
            print(f"File: {example_file_path} - Predicted Emotion: {predicted_emotion[0]}")
