import cv2
import numpy as np
import pytesseract
from tkinter import filedialog, messagebox
from matplotlib import pyplot as plt
import tkinter as tk
import re

# Specify the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def process_graph():
    try:
        # Open file dialog to select the graph image
        file_path = filedialog.askopenfilename(
            title="Select Graph Image",
            filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")]
        )

        if file_path:
            # Load the image
            img = cv2.imread(file_path)

            # Resize the image for better OCR
            height, width = img.shape[:2]
            new_width = 1600  # Higher width for improved resolution
            new_height = int((new_width / width) * height)
            img = cv2.resize(img, (new_width, new_height))

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply adaptive histogram equalization (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

            # Use morphological transformations to enhance text features
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) #one can alter these parameters
            morph = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

            # Use adaptive thresholding for better binarization
            #thresh = cv2.adaptiveThreshold(morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)  #original code
            _, thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Extract text with Tesseract
            custom_config = r'--oem 3 --psm 11'  
            extracted_text = pytesseract.image_to_string(thresh, config=custom_config)

            # Print extracted text
            print("Extracted Text:\n", extracted_text)

            # Dynamically process extracted text
            graph_data = process_extracted_text(extracted_text)
     
            print(f"Graph Data: {graph_data}")

            # Display the processed images
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 2, 1)
            plt.imshow(gray, cmap='gray')
            plt.title("Grayscale Image")

            plt.subplot(2, 2, 2)
            plt.imshow(enhanced, cmap='gray')
            plt.title("Enhanced Image (CLAHE)")

            plt.subplot(2, 2, 3)
            plt.imshow(morph, cmap='gray')
            plt.title("Morphologically Processed Image")

            plt.subplot(2, 2, 4)
            plt.imshow(thresh, cmap='gray')
            plt.title("Final Processed Image for OCR")

            plt.tight_layout()
            plt.show()

        else:
            messagebox.showwarning("No File Selected", "Please select a graph image file.")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def process_extracted_text(extracted_text):
    """
    Processes the extracted text to identify and extract graph data values only.
    """
    lines = extracted_text.split("\n")
    graph_data = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect numeric data (support floats and integers)
        values = [float(num) if '.' in num else int(num) for num in re.findall(r'\d+\.?\d*', line)]
        if values:
            graph_data.append(values)

    return graph_data

# Create the main tkinter window
root = tk.Tk()
root.title("Graph Extractor")
root.geometry("300x150")

# Create a button to trigger graph processing
process_button = tk.Button(root, text="Select Graph File", command=process_graph, font=("Arial", 14))
process_button.pack(pady=20)

# Start the tkinter main loop
root.mainloop()
