import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import cv2


# Fonction pour sélectionner l'image du film
def select_film_image():
    global film_path
    film_path = filedialog.askopenfilename(title="Sélectionnez l'image du film", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if film_path:
        messagebox.showinfo("Image sélectionnée", "L'image du film a bien été chargée !")

# Fonction pour sélectionner l'image de design
def select_design_image():
    global design_path
    design_path = filedialog.askopenfilename(title="Sélectionnez l'image de design", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if design_path:
        messagebox.showinfo("Image sélectionnée", "L'image de design a bien été chargée !")

# Fonction pour exécuter le traitement d'image et afficher le résultat
def process_and_display_image():
    if not film_path or not design_path:
        messagebox.showerror("Erreur", "Veuillez sélectionner les images avant de continuer.")
        return
    
    # Logique pour charger et traiter les images (simplifiée pour l'exemple)
    film_image = cv2.imread(film_path, cv2.IMREAD_COLOR)
    design_image = cv2.imread(design_path, cv2.IMREAD_GRAYSCALE)
    




# Charger les images du film de transfert DTF et de votre design


    facteur_reduct = 0.15  # Changer selon vos besoins

# Réduire le nombre de pixels de chaque image
    film_image= cv2.resize(film_image, None, fx=facteur_reduct, fy=facteur_reduct)
    design_image = cv2.resize(design_image, None, fx=facteur_reduct, fy=facteur_reduct)

# Initialiser l'extracteur de points de caractéristiques SIFT
    sift = cv2.SIFT_create()

# Trouver les points de caractéristiques et descripteurs pour les deux images
    film_keypoints, film_descriptors = sift.detectAndCompute(film_image, None)
    design_keypoints, design_descriptors = sift.detectAndCompute(design_image, None)

# Initialiser un objet BFMatcher avec le type de correspondance de distance euclidienne
    bf = cv2.BFMatcher()

# Trouver les correspondances entre les descripteurs du design et ceux du film de transfert DTF
    matches = bf.knnMatch(design_descriptors, film_descriptors, k=2)

# Garder les bonnes correspondances basées sur le ratio de Lowe
    good_matches = []
    for m, n in matches:
       if m.distance < 0.7 * n.distance:
           good_matches.append(m)

# Extraire les coordonnées des points de correspondance dans le film de transfert DTF
    film_points = np.float32([film_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    design_points = np.float32([design_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Trouver la transformation perspective entre les points du design et ceux du film de transfert DTF
    M, _ = cv2.findHomography(design_points, film_points, cv2.RANSAC)

# Appliquer la transformation perspective pour obtenir les coordonnées du design sur le film de transfert DTF
    h, w = design_image.shape
    design_on_film = cv2.perspectiveTransform(np.float32([[[0, 0]], [[0, h - 1]], [[w - 1, h - 1]], [[w - 1, 0]]]), M)

# Dessiner un rectangle autour du design sur le film de transfert DTF

    film_with_design_color = cv2.cvtColor(film_image, cv2.COLOR_BGR2RGB)
    cv2.polylines(film_with_design_color, [np.int32(design_on_film)], True, (0, 255, 0),30)
    display_image(film_with_design_color)
# Afficher l'image avec le design détecté
    #image2 = cv2.imshow('Design sur le film de transfert DTF', film_with_design_color)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #display_image(image2)


# Fonction pour afficher une image dans une nouvelle fenêtre Tkinter
def display_image(img):
    window = tk.Toplevel(app)
    window.title("Résultat du traitement")
    
    # Convertir l'image CV2 en un format que Tkinter peut afficher
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    main_window_width = app.winfo_width()
    main_window_height = app.winfo_height()
      
    
    # Calculer les nouvelles dimensions de l'image tout en maintenant les proportions
    img_width, img_height = img_pil.size
    ratio = 0.2
    new_width = int(img_width * ratio)
    new_height = int(img_height * ratio)

    img_pil_resized = img_pil.resize((new_width, new_height), Image.BICUBIC)
    img_tk = ImageTk.PhotoImage(image=img_pil_resized)
    window.geometry("800x600")
    
    
    # Afficher l'image avec un widget Label
    label = tk.Label(window, image=img_tk)
    
    label.image = img_tk
      
    label.pack(fill="both", expand=True)

    
      
    
    

# Configuration de la fenêtre principale
app = tk.Tk()
app.title("Détecteur de Design avec Tkinter")

# Configuration des boutons
frame = tk.Frame(app)
frame.pack(padx=10, pady=10)

btn_select_film_image = tk.Button(frame, text="Sélectionner l'image du film", command=select_film_image)
btn_select_film_image.pack(side=tk.LEFT, padx=5, pady=5)

btn_select_design_image = tk.Button(frame, text="Sélectionner l'image de design", command=select_design_image)
btn_select_design_image.pack(side=tk.LEFT, padx=5, pady=5)

btn_process_image = tk.Button(frame, text="Traiter les images", command=process_and_display_image)
btn_process_image.pack(side=tk.LEFT, padx=5, pady=5)

app.mainloop()