from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        CSVLogger, EarlyStopping)
from model import get_model
import argparse
from datasets import ECGSequence

if __name__ == "__main__":
    # Obtine datele si antreneaza
    parser = argparse.ArgumentParser(description='Antreneaza reteaua neurala.')
    parser.add_argument('path_to_hdf5', type=str,
                        help='calea catre fisierul hdf5 care contine traseele')
    parser.add_argument('path_to_csv', type=str,
                        help='calea catre fisierul csv care contine adnotarile')
    parser.add_argument('--val_split', type=float, default=0.02,
                        help='numar intre 0 si 1 care determina cat din date sa fie folosit pentru validare. Restul este folosit pentru antrenare. Implicit: 0.02')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='numele setului de date hdf5 care contine traseele')
    args = parser.parse_args()
    
    # Setari pentru optimizare
    loss = 'binary_crossentropy'
    lr = 0.00001
    batch_size = 64
    opt = Adam(lr)
    
    # Definirea callback-urilor
    callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=7,
                                   min_lr=lr / 100),
                 EarlyStopping(patience=9,  # Patience trebuie sa fie mai mare decat cel din ReduceLROnPlateau
                               min_delta=0.00001)]

    # Obtine secventele de antrenare si validare
    train_seq, valid_seq = ECGSequence.get_train_and_val(
        args.path_to_hdf5, args.dataset_name, args.path_to_csv, batch_size, args.val_split)

    # Daca se continua o sectiune intrerupta, decomentati linia de mai jos:
    #   model = keras.models.load_model(PATH_TO_PREV_MODEL, compile=False)
    
    # Crearea modelului
    model = get_model(train_seq.n_classes)
    model.compile(loss=loss, optimizer=opt)
    
    # Crearea jurnalului
    callbacks += [TensorBoard(log_dir='./logs', write_graph=False),
                  CSVLogger('training.log', append=False)]  # Schimbati append la true daca continuati antrenarea
    
    # Salvarea celui mai bun si ultimului model
    callbacks += [ModelCheckpoint('./backup_model_last.keras'),
                  ModelCheckpoint('./backup_model_best.keras', save_best_only=True)]
    
    # Antrenarea retelei neurale
    history = model.fit(train_seq,
                        epochs=70,
                        initial_epoch=0,  # Daca continuati o sectiune intrerupta, modificati aici
                        callbacks=callbacks,
                        validation_data=valid_seq,
                        verbose=1)
    
    # Salvarea rezultatului final
    model.save("./final_model.keras")
