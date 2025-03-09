import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from datasets import ECGSequence  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Obține performanța pe setul de testare dintr-un fișier hdf5')
    parser.add_argument('path_to_hdf5', type=str,
                        help='calea către fișierul hdf5 care conține traseele')
    parser.add_argument('path_to_model',  # sau model_date_order.hdf5
                        help='fișierul care conține modelul antrenat.')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='numele setului de date hdf5 care conține traseele')
    parser.add_argument('--output_file', default="./dnn_output.npy",  # sau predictions_date_order.csv
                        help='fișierul de ieșire CSV.')
    parser.add_argument('-bs', type=int, default=32,
                        help='Dimensiunea lotului.')

    args, unk = parser.parse_known_args()
    if unk:
        warnings.warn("Argumente necunoscute:" + str(unk) + ".")

    # Importarea datelor
    seq = ECGSequence(args.path_to_hdf5, args.dataset_name, batch_size=args.bs)
    
    # Importarea modelului
    model = load_model(args.path_to_model, compile=False)
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    
    # Calcularea scorurilor de predicție
    y_score = model.predict(seq, verbose=1)

    # Salvarea rezultatelor
    np.save(args.output_file, y_score)

    print("Predicțiile de ieșire au fost salvate")
