import cv2
import numpy as np
import pytesseract
from tkinter import filedialog, messagebox
from matplotlib import pyplot as plt
import tkinter as tk
import re

# Specify the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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
            new_width = 1200
            new_height = int((new_width / width) * height)
            img = cv2.resize(img, (new_width, new_height))

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply adaptive histogram equalization (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Use edge detection to enhance text boundaries
            edges = cv2.Canny(enhanced, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(edges, kernel, iterations=1)

            # Combine dilated edges with the enhanced image
            combined = cv2.addWeighted(enhanced, 0.7, dilated, 0.3, 0)

            # Apply thresholding for OCR
            _, thresh = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Use Tesseract to extract text (mixed text and numbers)
            custom_config = r'--oem 3 --psm 11'  # PSM 11 for sparse text
            extracted_text = pytesseract.image_to_string(thresh, config=custom_config)

            print("Extracted Text:\n", extracted_text)

            # Dynamically process extracted text
            title, x_axis, y_axis, data_values = process_extracted_text(extracted_text)

            print(f"Title: {title}")
            print(f"X-Axis: {x_axis}")
            print(f"Y-Axis: {y_axis}")
            print(f"Data Values: {data_values}")

            # Display the processed images
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 2, 1)
            plt.imshow(gray, cmap='gray')
            plt.title("Grayscale Image")

            plt.subplot(2, 2, 2)
            plt.imshow(enhanced, cmap='gray')
            plt.title("Enhanced Image (CLAHE)")

            plt.subplot(2, 2, 3)
            plt.imshow(dilated, cmap='gray')
            plt.title("Edges (Dilated)")

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
    Dynamically processes the extracted text to identify the title, axis labels, and data values.
    """
    lines = extracted_text.split("\n")
    title = None
    x_axis = None
    y_axis = None
    data_values = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Assume the first non-empty line is the title
        if title is None:
            title = line
            continue

        # Detect axis labels dynamically
        if re.search(r"[A-Za-z]+", line) and not re.search(r"\d", line):
            if x_axis is None:
                x_axis = line  # First textual line after title is X-axis labels
            elif y_axis is None:
                y_axis = line  # Second textual line is Y-axis labels
            continue

        # Detect numeric data (support floats and integers)
        values = [float(num) if '.' in num else int(num) for num in re.findall(r'\d+\.?\d*', line)]
        if values:
            data_values.append(values)

    return title, x_axis, y_axis, data_values

# Create the main tkinter window
root = tk.Tk()
root.title("Graph Extractor")
root.geometry("300x150")

# Create a button to trigger graph processing
process_button = tk.Button(root, text="Select Graph File", command=process_graph, font=("Arial", 14))
process_button.pack(pady=20)

# Start the tkinter main loop
root.mainloop()
