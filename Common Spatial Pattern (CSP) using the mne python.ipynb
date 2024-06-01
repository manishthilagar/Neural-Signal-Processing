import numpy as np
import mne
from mne.datasets import sample
from mne.decoding import CSP
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load the sample dataset
data_path = sample.data_path()
raw = mne.io.read_raw_fif(data_path + '/MEG/sample/sample_audvis_raw.fif', preload=True)

# Select channels of interest (EEG channels)
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)

# Set the events and event_id
events = mne.find_events(raw, stim_channel='STI 014')
event_id = {'left/auditory': 1, 'right/auditory': 2}

# Create epochs around events
epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.5, picks=picks, baseline=(None, 0), preload=True)
labels = epochs.events[:, -1]

# Extract data and labels
X = epochs.get_data()
y = labels

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CSP
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

# Initialize a classifier
svm = SVC(kernel='linear', C=1)

# Create a pipeline
clf = Pipeline([('CSP', csp), ('SVM', svm)])

# Train the classifier
clf.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
print("Classification report:\n", classification_report(y_test, y_pred))
print("Accuracy score:", accuracy_score(y_test, y_pred))
