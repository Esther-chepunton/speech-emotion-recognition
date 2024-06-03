import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import os

# Load audio files
def load_data(file_paths):
    data = []
    labels = []
    for file_path in file_paths:
        print(file_path)  # Check the file path
        try:
            # Load audio file
            y, sr = librosa.load(file_path, res_type='kaiser_fast')

            # Extract features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfccs_processed = np.mean(mfccs.T,axis=0)

            

            # Add to data and labels lists
            data.append(mfccs_processed)
            if 'angry' in file_path:
                labels.append(0)
            elif 'disgust' in file_path:
                labels.append(1)
            elif 'fear' in file_path:
                labels.append(2)
            elif 'happy' in file_path:
                labels.append(3)
            elif 'neutral' in file_path:
                labels.append(4)
            elif 'ps' in file_path:
                labels.append(5)
            elif 'sad' in file_path:
                labels.append(6)
            elif 'surprise' in file_path:
                labels.append(7)
            
            else:
                raise Exception(f"Invalid file path: {file_path}")
        except Exception as e:
            print(f"Error loading file: {file_path} - {str(e)}")

    return np.array(data), np.array(labels)

# Load all audio files in a directory
root_dir = "C:/Users/Esther/Desktop/speech-emotion/archive (2)/TESS Toronto emotional speech set data"
file_paths = []
for emotion in os.listdir(root_dir):
    emotion_dir = os.path.join(root_dir, emotion)
    for file in os.listdir(emotion_dir):
        file_path = os.path.join(emotion_dir, file)
        file_paths.append(file_path)

data, labels = load_data(file_paths)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train SVM classifier
clf = SVC(kernel='linear', random_state=42)
clf.fit(X_train, y_train)

# Evaluate classifier on testing data
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
