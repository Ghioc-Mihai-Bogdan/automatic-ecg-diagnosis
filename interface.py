import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import subprocess
import threading
import numpy as np
from datetime import datetime
from fpdf import FPDF
from PIL import Image, ImageTk

class Interfata:
    def __init__(self, master):
        self.master = master
        self.master.title("ToraxInsight")
        self.master.geometry("1800x720")  # Setare rezolutie 1280x720
        self.master.resizable(False, False)
        
        # Adaugare imagine de fundal
        self.background_image = Image.open("./background.jpg")
        self.background_photo = ImageTk.PhotoImage(self.background_image)
        
        self.background_label = tk.Label(master, image=self.background_photo)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Variabila pentru stocarea caii fisierului selectat
        self.pacient = None
        
        # Titlu in centru-sus
        self.titlu_label = tk.Label(master, text="ToraxInsight: aplicatie pentru diagnosticare", font=("Helvetica", 30))
        self.titlu_label.pack(pady=10)

        # Buton pentru setari avansate
        self.setari_avansate_button = tk.Button(master, text="Setari avansate", font=("Helvetica", 12), command=self.toggle_setari_avansate, width=20, height=2)
        self.setari_avansate_button.pack(pady=10)

        # Frame pentru setari avansate
        self.setari_avansate_frame = tk.Frame(master)
        self.setari_avansate_frame.pack(pady=10)
        self.setari_avansate_frame.pack_forget()  # Ascunde initial

        # Buton pentru selectarea modelului
        self.model_label = tk.Label(self.setari_avansate_frame, text="Selecteaza modelul:", font=("Helvetica", 12))
        self.model_button = tk.Button(self.setari_avansate_frame, text="Alege modelul", command=self.adauga_model)
        
        self.model_label.grid(row=0, column=0, padx=10, pady=5)
        self.model_button.grid(row=0, column=1, padx=10, pady=5)

        # Variabila pentru stocarea caii modelului selectat
        self.model_path = "./modele/model.hdf5"
        
        # Butoane pentru selectarea pragului de probabilitate
        self.probabilitate_label = tk.Label(self.setari_avansate_frame, text="Selecteaza pragul de probabilitate:", font=("Helvetica", 12))
        self.probabilitate_label.grid(row=1, column=0, padx=10, pady=5)
        
        self.probabilitate_var = tk.DoubleVar(value=0.5)
        
        self.praguri_frame = tk.Frame(self.setari_avansate_frame)
        self.praguri_frame.grid(row=1, column=1, padx=10, pady=5)
        
        self.prag_mica_button = tk.Radiobutton(self.praguri_frame, text="Mica (0.5)", variable=self.probabilitate_var, value=0.5)
        self.prag_medie_button = tk.Radiobutton(self.praguri_frame, text="Medie (0.6)", variable=self.probabilitate_var, value=0.6)
        self.prag_mare_button = tk.Radiobutton(self.praguri_frame, text="Mare (0.7)", variable=self.probabilitate_var, value=0.7)
        
        self.prag_mica_button.pack(side=tk.LEFT, padx=5)
        self.prag_medie_button.pack(side=tk.LEFT, padx=5)
        self.prag_mare_button.pack(side=tk.LEFT, padx=5)

        # Frame pentru partea stanga-mijloc
        frame_stanga = tk.Frame(master)
        frame_stanga.pack(side=tk.LEFT, padx=20, pady=10)

        # Text "Explicatie" deasupra
        self.explicatie_label = tk.Label(frame_stanga, text="Adauga fisierul cu analizele pacientului apasand butonul de mai jos.", font=("Helvetica", 14))
        self.explicatie_label.grid(row=0, column=0, pady=30)
        
        # Adauga fisier hdf5
        self.adauga_fisier_button = tk.Button(frame_stanga, text="Adauga fisier", font=("Helvetica", 12), command=self.adauga_fisier, width=20, height=3)
        self.adauga_fisier_button.grid(row=1, column=0, pady=10)
        
        # Label pentru afisarea caii fisierului
        self.cale_label = tk.Label(frame_stanga, text="", font=("Helvetica", 12), wraplength=400)
        
        # Buton de procesare diagnostic
        self.buton = tk.Button(frame_stanga, text="Proceseaza analizele", font=("Helvetica", 12), width=20, height=3, command=self.start_procesare)
        self.buton.grid(row=3, column=0, pady=10)
        
        # Bara de progres
        self.progress = ttk.Progressbar(frame_stanga, orient=tk.HORIZONTAL, length=400, mode='indeterminate')
        
        # Label pentru afisarea mesajului de procesare finalizata
        self.mesaj_procesare_label = tk.Label(frame_stanga, text="", font=("Helvetica", 12))
        
        # Frame pentru partea dreapta
        self.frame_dreapta = tk.Frame(master)
        self.frame_dreapta.pack(side=tk.RIGHT, padx=20, pady=10)
        
        # Titlu pentru tabel
        self.tabel_titlu = tk.Label(self.frame_dreapta, text="Rezultatele analizelor", font=("Helvetica", 16))
        self.tabel_titlu.pack(pady=10)

        # Tabel pentru rezultate
        self.tabel = ttk.Treeview(self.frame_dreapta, columns=("Afectiune", "Rezultat"), show='headings')
        self.tabel.heading("Afectiune", text="Afectiune")
        self.tabel.heading("Rezultat", text="Rezultat")
        self.tabel.pack()

        # Afectiuni
        self.afectiuni = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]
        self.prag = 0.5  # Prag pentru pozitiv/negativ
        
        # Buton pentru printare analize
        self.print_button = tk.Button(self.frame_dreapta, text="Printeaza analize", font=("Helvetica", 12), command=self.deschide_fereastra_printare)
        self.print_button.pack(pady=10)
        self.print_button.config(state=tk.DISABLED)  # Dezactivează inițial butonul

    def toggle_setari_avansate(self):
        if self.setari_avansate_frame.winfo_ismapped():
            self.setari_avansate_frame.pack_forget()
        else:
            self.setari_avansate_frame.pack(pady=10)

    def adauga_fisier(self):
        file_path = filedialog.askopenfilename(filetypes=[("Fisiere HDF5", "*.h5;*.hdf5")])
        if file_path:
            self.pacient = file_path
            self.cale_label.config(text=f'Calea catre fisier: "{self.pacient}"')
            self.cale_label.grid(row=2, column=0, pady=5)
            print(f"Calea fisierului selectat: {self.pacient}")
            self.mesaj_procesare_label.config(text="")  # Reseteaza mesajul de eroare

    def adauga_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("Fisiere HDF5", "*.h5;*.hdf5")])
        if model_path:
            self.model_path = model_path
            self.model_button.config(text=f"Model selectat: {model_path.split('/')[-1]}")
            print(f"Calea modelului selectat: {self.model_path}")
            self.mesaj_procesare_label.config(text="")  # Reseteaza mesajul de eroare
    
    def start_procesare(self):
        if self.pacient:
            if self.model_path:
                self.progress.grid(row=4, column=0, pady=10, columnspan=2)
                self.mesaj_procesare_label.grid_forget()  # Ascunde mesajul de procesare anterioara
                self.progress.start()
                threading.Thread(target=self.proceseaza_analizele).start()
            else:
                self.mesaj_procesare_label.grid(row=5, column=0, pady=5)
                self.mesaj_procesare_label.config(text="Selectati un model pentru procesare.")
        else:
            self.mesaj_procesare_label.grid(row=5, column=0, pady=5)
            self.mesaj_procesare_label.config(text="Selectati un fisier pentru procesare.")

    def proceseaza_analizele(self):
        script_path = './predict.py'
        output_path = "./Rezultate/rezultate_pacient.txt"
        
        try:
            # Convertirea fisierului HDF5 in fisier text folosind scriptul predict.py
            subprocess.run(["python", script_path, self.pacient, self.model_path, output_path], check=True)
            print("Conversia a fost finalizata.")
            self.progress.stop()
            self.progress.grid_forget()
            # Incarca rezultatele si actualizeaza tabelul
            self.incarca_si_actualizeaza_rezultate()
        except subprocess.CalledProcessError as e:
            print(f"Conversia a esuat. Eroare: {e}")
            self.progress.stop()
            self.progress.grid_forget()
            self.mesaj_procesare_label.grid(row=5, column=0, pady=5)
            self.mesaj_procesare_label.config(text="Procesarea a esuat. Verificati consola pentru detalii.")

    def incarca_si_actualizeaza_rezultate(self):
        file_path_txt = './Rezultate/rezultate_pacient.txt'
        try:
            # Citirea datelor din fisierul text
            with open(file_path_txt, 'r') as file:
                line = file.readline().strip()
            
            # Transformam linia intr-o lista de valori
            values = [float(val) for val in line.split()]
            
            # Verificam daca avem suficiente valori
            if len(values) != len(self.afectiuni):
                print(f"Numarul de valori din fisierul text ({len(values)}) nu corespunde cu numarul de afectiuni ({len(self.afectiuni)})")
                return
            
            self.tabel.delete(*self.tabel.get_children())  # Sterge datele vechi din tabel
            
            # Parcurgem fiecare valoare si adaugam valorile in tabel
            for i, valoare in enumerate(values):
                afectiune = self.afectiuni[i]
                rezultat = "Pozitiv" if valoare >= self.probabilitate_var.get() else "Negativ"
                self.tabel.insert("", "end", values=(afectiune, rezultat))
            
            # Activăm butonul de printare analize după ce tabelul a fost actualizat
            self.print_button.config(state=tk.NORMAL)
        
        except Exception as e:
            print(f"Eroare la incarcarea fisierului: {e}")
    
    def deschide_fereastra_printare(self):
        fereastra_printare = tk.Toplevel(self.master)
        fereastra_printare.title("Completeaza date pacient")
        fereastra_printare.geometry("400x500")

        tk.Label(fereastra_printare, text="Nume:").pack(pady=5)
        nume_entry = tk.Entry(fereastra_printare)
        nume_entry.pack(pady=5)

        tk.Label(fereastra_printare, text="Prenume:").pack(pady=5)
        prenume_entry = tk.Entry(fereastra_printare)
        prenume_entry.pack(pady=5)

        tk.Label(fereastra_printare, text="CNP:").pack(pady=5)
        cnp_entry = tk.Entry(fereastra_printare)
        cnp_entry.pack(pady=5)

        tk.Label(fereastra_printare, text="Telefon:").pack(pady=5)
        telefon_entry = tk.Entry(fereastra_printare)
        telefon_entry.pack(pady=5)

        tk.Label(fereastra_printare, text="Adresa:").pack(pady=5)
        adresa_entry = tk.Entry(fereastra_printare)
        adresa_entry.pack(pady=5)

        varsta_label = tk.Label(fereastra_printare, text="Varsta: ")
        varsta_label.pack(pady=5)

        sex_label = tk.Label(fereastra_printare, text="Sex: ")
        sex_label.pack(pady=5)

        # Actualizare automata a varstei si sexului pe baza CNP-ului
        def actualizeaza_varsta_sex(event):
            cnp = cnp_entry.get()
            if len(cnp) == 13 and cnp.isdigit():
                try:
                    sex = int(cnp[0])
                    an = int(cnp[1:3])
                    luna = int(cnp[3:5])
                    zi = int(cnp[5:7])
                    
                    if sex in [1, 2]:
                        an += 1900
                    elif sex in [3, 4]:
                        an += 1800
                    elif sex in [5, 6]:
                        an += 2000
                    
                    data_nastere = datetime(an, luna, zi)
                    varsta = (datetime.now() - data_nastere).days // 365
                    sex_str = "Masculin" if sex % 2 == 1 else "Feminin"
                    
                    varsta_label.config(text=f"Varsta: {varsta}")
                    sex_label.config(text=f"Sex: {sex_str}")
                except Exception as e:
                    varsta_label.config(text="Varsta: Invalid")
                    sex_label.config(text="Sex: Invalid")
            else:
                varsta_label.config(text="Varsta: Invalid")
                sex_label.config(text="Sex: Invalid")

        cnp_entry.bind("<KeyRelease>", actualizeaza_varsta_sex)

        def finalizare():
            nume = nume_entry.get()
            prenume = prenume_entry.get()
            cnp = cnp_entry.get()
            telefon = telefon_entry.get()
            adresa = adresa_entry.get()
            varsta = varsta_label.cget("text").split(": ")[1]
            sex = sex_label.cget("text").split(": ")[1]
            
            self.genereaza_fisa(nume, prenume, cnp, telefon, adresa, varsta, sex)
            fereastra_printare.destroy()

        finalizare_button = tk.Button(fereastra_printare, text="Finalizare", command=finalizare)
        finalizare_button.pack(pady=20)

    def genereaza_fisa(self, nume, prenume, cnp, telefon, adresa, varsta, sex):
        file_name = "./Rezultate/fisa_analize_pacient.pdf"  # Numele de fisier static

        pdf = FPDF()
        pdf.add_page()

        # Adauga fontul DejaVuSans fara encoding
        pdf.add_font("Roboto", fname="./font/Roboto-Black.ttf", uni=True)

        # Seteaza fontul fara encoding
        pdf.set_font("Roboto", size=12)

        pdf.cell(200, 10, txt="Fisa Analize Pacient", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Nume: {nume}", ln=True, align='L')
        pdf.cell(200, 10, txt=f"Prenume: {prenume}", ln=True, align='L')
        pdf.cell(200, 10, txt=f"CNP: {cnp}", ln=True, align='L')
        pdf.cell(200, 10, txt=f"Telefon: {telefon}", ln=True, align='L')
        pdf.cell(200, 10, txt=f"Adresa: {adresa}", ln=True, align='L')
        pdf.cell(200, 10, txt=f"Varsta: {varsta}", ln=True, align='L')
        pdf.cell(200, 10, txt=f"Sex: {sex}", ln=True, align='L')

        pdf.cell(200, 10, txt="Rezultatele analizelor:", ln=True, align='L')
        for item in self.tabel.get_children():
            values = self.tabel.item(item, "values")
            pdf.cell(200, 10, txt=f"{values[0]}: {values[1]}", ln=True, align='L')

        pdf.output(file_name)
        print(f"Fisa de analize a fost generata cu succes si a suprascris fisierul existent.")

def main():
    root = tk.Tk()
    interfata = Interfata(root)
    root.mainloop()

if __name__ == "__main__":
    main()
