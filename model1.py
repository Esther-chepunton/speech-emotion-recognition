import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load audio files
def load_data(file_paths):
    data = []
    file = []
    for file_path in file_paths:
        # Load audio file
        y, sr = librosa.load(file_path,sr=None)

        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T,axis=0)

        # Add to data and labels lists
        data.append(mfccs_processed)
        if 'angry' in file_path:
            file.append(0)
        elif 'disgust' in file_path:
            file.append(1)
        elif 'fear' in file_path:
            file.append(2)
        elif 'happy' in file_path:
            file.append(3)
        elif 'neutral' in file_path:
            file.append(4)
        elif 'sad' in file_path:
            file.append(5)
        else:
            raise Exception(f"Invalid file path: {file_path}")

    return np.array(data), np.array(file)

# Load all audio files in a directory
root_dir = r"C:\Users\Esther\Desktop\speech-emotion\data\wav"
file_paths = []
if os.path.isdir(root_dir):
    file_paths = [os.path.join(root_dir, emotion, file) for emotion in os.listdir(root_dir) for file in os.listdir(os.path.join(root_dir, emotion))]
else:
    print("Directory exists")
data, file = load_data(file_paths)

# Split data into training and testing sets
if len(data) > 0 and len(file) > 0:
    X_train, X_test, y_train, y_test = train_test_split(data, file, test_size=0.2, random_state=42)
    
    # Train SVM classifier (support vector Machine)for classification and linear regression
    clf = SVC(kernel='linear', random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate classifier on testing data
    accuracy = clf.score(X_test, y_test)
    print("Accuracy: {:.2f}".format(accuracy))
else:
    print("No data found")
