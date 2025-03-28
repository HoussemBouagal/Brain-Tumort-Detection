# Brain Tumor Detection

This project is a **Brain Tumor Detection** system using a deep learning model and a graphical user interface (GUI) built with Tkinter.

## Features
- **Deep Learning Model**: Uses a pre-trained TensorFlow model for brain tumor classification.
- **User-Friendly Interface**: Built with Tkinter, featuring dark mode UI.
- **Tumor Classification**: Detects and classifies brain tumors into four categories: Glioma, Meningioma, No Tumor, and Pituitary.
- **Image Upload & Display**: Allows users to upload an image for analysis.
- **Confidence Score**: Displays the prediction confidence percentage.
- **Developer Info**: Shows details about the development team.

## Requirements
- Python 3.8+
- TensorFlow
- NumPy
- Pillow
- Tkinter (built-in with Python)

## Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/your-repo/brain-tumor-detection.git
   cd brain-tumor-detection
   ```
2. Install dependencies:
   ```sh
   pip install tensorflow numpy pillow
   ```
3. Run the application:
   ```sh
   python main.py
   ```

## Usage
1. Click on **Load Image** to upload a brain scan image.
2. The system will analyze the image and display the predicted tumor type along with the confidence percentage.
3. The uploaded image will be displayed in the interface.
4. Click on **Dev by** to view developer information.

## File Structure
```
brain-tumor-detection/
│── brain_tumor_model.keras  # Pre-trained model file
│── brain-cancer.png         # Icon for the application
│── main.py                  # Main application script
│── README.md                # Project documentation
```

## Notes
- Ensure that `brain_tumor_model.keras` is present in the same directory as `main.py`.
- If the icon file is missing, the application will still run, but without a custom icon.

## Developer
- **Bouagal Houssem Eddine**

## License
This project is licensed under the MIT License.

