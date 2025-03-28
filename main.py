import os
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, messagebox
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageTk

# Auto-detect model path
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "brain_tumor_model.keras")
icon_path = os.path.join(script_dir, "brain-cancer.png")  # تحديد مسار الأيقونة

# Load the trained model
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Model classes
class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Function to open an image
def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.png;*.jpeg")])
    if file_path:
        result_label.config(text="🔄 Analyzing...", fg="#F8C471")
        root.update_idletasks()
        predict_image(file_path)

# Function to predict and display result
def predict_image(img_path):
    img = Image.open(img_path).resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Model prediction
    prediction = model.predict(img_array)
    confidence = np.max(prediction) * 100  # Get confidence percentage
    predicted_class = class_names[np.argmax(prediction)]
    
    # Display results
    display_image(img_path)
    result_label.config(text=f"🧠 Tumor Detected: {predicted_class}\nConfidence: {confidence:.2f}%", fg="#4CAF50")

# Function to display selected image
def display_image(img_path):
    img = Image.open(img_path).resize((250, 250))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

# Function to show team info
def show_team():
    team_members = "Development by:\n- Bouagal Houssem Eddine"
    messagebox.showinfo("Development Team", team_members)

# Create Tkinter window
root = tk.Tk()
root.title("Brain Tumor Detection")
root.geometry("450x550")
root.resizable(False, False)
root.configure(bg="#222831")  # Dark mode background

# Set window icon
if os.path.exists(icon_path):
    icon_image = Image.open(icon_path)
    icon_photo = ImageTk.PhotoImage(icon_image)
    root.iconphoto(False, icon_photo)  # تعيين الأيقونة
else:
    print(f"⚠️ Warning: Icon file not found at {icon_path}")

# Styling
btn_style = {"font": ("Arial", 12, "bold"), "bg": "#00ADB5", "fg": "white", "bd": 0, "relief": "flat", "padx": 10, "pady": 5}
label_style = {"font": ("Arial", 14, "bold"), "fg": "#EEEEEE", "bg": "#222831"}

# Create a frame for layout
frame = Frame(root, bg="#222831")
frame.pack(pady=20)

header_label = tk.Label(frame, text="Brain Tumor Detection", font=("Arial", 18, "bold"), bg="#222831", fg="white", pady=10)
header_label.pack(fill=tk.X, pady=10)

# Load Image Button
btn_load = Button(frame, text="📂 Load Image", command=open_image, **btn_style)
btn_load.pack(pady=10)

# Image display area
image_label = Label(frame, bg="#393E46")
image_label.pack(pady=10)

# Prediction result label
result_label = Label(frame, text="", **label_style)
result_label.pack(pady=20)

# Create button frame
button_frame = Frame(root, bg="#222831")
button_frame.pack(pady=10)

# Team button
team_button = Button(button_frame, text="Dev by", font=("Arial", 12), bg="#FFC107", fg="black", command=show_team)
team_button.pack()

# Run the Tkinter app
root.mainloop()
