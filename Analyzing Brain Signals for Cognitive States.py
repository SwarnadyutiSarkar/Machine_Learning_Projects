import mne
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load and preprocess EEG data
raw = mne.io.read_raw_brainvision('eeg_data.vhdr', preload=True)
events = mne.find_events(raw, stim_channel='STI 014')

# Extract epochs
event_id = {'attention': 1, 'meditation': 2}
epochs = mne.Epochs(raw, events, event_id, tmin=-0.5, tmax=2, baseline=(None, 0), preload=True)

# Prepare features and labels
X = epochs.get_data()
y = epochs.events[:, -1]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))
