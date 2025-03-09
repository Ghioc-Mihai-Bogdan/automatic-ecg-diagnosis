import h5py
import numpy as np

# Calea către fișierul HDF5 original
input_file_path = "C:\\Users\\Mihai\\Desktop\\Licenta\\data\\data\\ecg_tracings.hdf5"

# Deschidem fișierul original pentru citire
with h5py.File(input_file_path, 'r') as f:
    # Extragem dataset-ul "tracings"
    tracings = f['tracings'][:]

# Salvăm datele pentru fiecare dintre primii 80 pacienți în fișiere separate
for i in range(80):
    patient_tracing = tracings[i:i+1, :, :]
    patient_file_path = f"./pacienti/pacient_{i+1}.hdf5"
    with h5py.File(patient_file_path, 'w') as f_patient:
        f_patient.create_dataset('tracings', data=patient_tracing)

# Salvăm datele pentru ceilalți pacienți într-un singur fișier HDF5 pe partitia D
remaining_patients_tracings = tracings[80:, :, :]
output_file_path = "D:\\remaining_patients_tracings.hdf5"
with h5py.File(output_file_path, 'w') as f_output:
    f_output.create_dataset('tracings', data=remaining_patients_tracings)

print("Datele pentru primii 80 pacienți au fost salvate cu succes în fișiere HDF5 separate.")
print(f"Datele pentru ceilalți pacienți au fost salvate cu succes în fișierul: {output_file_path}")
