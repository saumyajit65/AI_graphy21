import cv2
import numpy as np
import pytesseract
from tkinter import filedialog, messagebox
from matplotlib import pyplot as plt
import tkinter as tk
import re
from difflib import SequenceMatcher

# Specify the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def remove_duplicates(text_list, similarity_threshold=0.85):
    """
    Removes duplicate strings from the list based on a similarity threshold.
    """
    unique_texts = []
    for text in text_list:
        if not any(SequenceMatcher(None, text, existing).ratio() > similarity_threshold for existing in unique_texts):
            unique_texts.append(text)
    return unique_texts

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

            # Use morphological transformations to clean noise and enhance text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

            # Use edge detection to enhance text boundaries
            edges = cv2.Canny(enhanced, 50, 150)
            dilated = cv2.dilate(edges, kernel, iterations=1)

            # Combine dilated edges with the enhanced image
            combined = cv2.addWeighted(enhanced, 0.7, dilated, 0.3, 0)

            # Generate multiple thresholded images for OCR
            _, thresh_morph = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, thresh_combined = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Use Tesseract to extract text (mixed text and numbers)
            custom_config = r'--oem 3 --psm 11'  # PSM 11 for sparse text
            extracted_text_morph = pytesseract.image_to_string(thresh_morph, config=custom_config)
            extracted_text_combined = pytesseract.image_to_string(thresh_combined, config=custom_config)

            # Combine and deduplicate extracted text
            all_text = [extracted_text_morph, extracted_text_combined]
            unique_text = remove_duplicates(all_text)

            print("Unique Extracted Text:")
            for idx, text in enumerate(unique_text, 1):
                print(f"Method {idx}:\n{text}\n")

            # Dynamically process extracted text
            graph_data = []
            for text in unique_text:
                graph_data.extend(process_extracted_text(text))

            print(f"Graph Data: {graph_data}")

            # Display the processed images
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 3, 1)
            plt.imshow(gray, cmap='gray')
            plt.title("Grayscale Image")

            plt.subplot(2, 3, 2)
            plt.imshow(enhanced, cmap='gray')
            plt.title("Enhanced Image (CLAHE)")

            plt.subplot(2, 3, 3)
            plt.imshow(morph, cmap='gray')
            plt.title("Morphologically Processed Image")

            plt.subplot(2, 3, 4)
            plt.imshow(combined, cmap='gray')
            plt.title("Combined Image")

            plt.subplot(2, 3, 5)
            plt.imshow(thresh_combined, cmap='gray')
            plt.title("Thresholded Combined Image")

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
