This project uses deep learning to classify whether a subject has knee osteoarthritis (OA) based on their gait waveforms captured from markerless motion capture (Theia3D). It processes frontal (Y) and sagittal (X) plane joint angle waveforms from a single leg per subject and trains a neural network for binary classification: 0 = Healthy, 1 = Medial OA

Notes:
- Data was preprocessed in R and exported to CSV before modeling in Python.
- Currently, only one leg per subject is used to avoid duplication bias..
