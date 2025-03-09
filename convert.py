import numpy as np

# Specifică calea către fișierul .npy
file_path_npy = r'./Rezultate/pacient_analizat.npy'

# Încarcă fișierul
data = np.load(file_path_npy)

# Specifică calea pentru fișierul .txt
file_path_txt = r'./Rezultate/rezultate_pacient.txt'

# Salvează datele în fișierul .txt
np.savetxt(file_path_txt, data, fmt='%s')

print(f"Datele au fost salvate în {file_path_txt}")

