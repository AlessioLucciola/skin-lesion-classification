import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from config import NUM_DROPOUT_LAYERS, PATH_TO_SAVE_RESULTS, NUM_CLASSES, DROPOUT_P
from models.GradCAM import GradCAM
from models.MSLANet import MSLANet
from utils.utils import select_device

device = select_device()
SAM_IMG_SIZE = 128
KEEP_BACKGROUND = False
DEMO_MODEL_PATH = "MSLANet_2024-01-19_18-11-26"
DEMO_MODEL_EPOCH = "86"

def get_model(model_path, epoch):
    model = MSLANet(num_classes=NUM_CLASSES, dropout_num=NUM_DROPOUT_LAYERS, dropout_p=DROPOUT_P).to(device)

    state_dict = torch.load(
        f"{PATH_TO_SAVE_RESULTS}/{model_path}/models/melanoma_detection_{epoch}.pt", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    return model

def decode_prediction(pred):
    if pred == 0:
        return tuple(("Melanocytic nevi", 0))
    elif pred == 1:
        return tuple(("Benign lesions of the keratosis", 0))
    elif pred == 2:
        return tuple(("Melanoma", 1))
    elif pred == 3:
        return tuple(("Actinic keratoses and intraepithelial carcinoma", 1))
    elif pred == 4:
        return tuple(("Basal cell carcinoma", 1))
    elif pred == 5:
        return tuple(("Dermatofibroma", 0))
    else:
        return tuple(("Vascular lesion", 0))

def get_gradcam_output(image):
    cam_instance = GradCAM()
    thresholds = [70, 110]
    ret = []
    for t in thresholds:
        out = cam_instance.generate_cam(image=image, threshold=t)
        ret.append(out)
    return ret

def process_image(image_path, model_path=DEMO_MODEL_PATH, epoch=DEMO_MODEL_EPOCH):
    model = get_model(model_path, epoch)
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Open and preprocess the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    # Make a prediction using the model
    with torch.no_grad():
        output = model(image)

    probabilities = torch.softmax(output, dim=1)[0]
    predicted_class = torch.argmax(probabilities).item()

    grad_cam_output = get_gradcam_output(image)

    return image, grad_cam_output, predicted_class

def open_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])
    if file_path:
        processed_image, gradcam_output, predicted_class = process_image(file_path)

        # Display the selected image and the predicted class
        display_image(processed_image, gradcam_output, predicted_class)

def display_image(image, grad_cam_output, predicted_class):
    # Convert the PyTorch tensor to a NumPy array
    image_one = image.cpu().squeeze(0).numpy().transpose((1, 2, 0))
    image_two = grad_cam_output[0][1].cpu().squeeze(0).numpy().transpose((1, 2, 0)) #cropped img thr1
    image_three = grad_cam_output[1][1].cpu().squeeze(0).numpy().transpose((1, 2, 0)) #cropped img thr2
    heatmap = grad_cam_output[0][0].cpu().squeeze(0).numpy().transpose((1, 2, 0))
    rect_one = grad_cam_output[0][2].cpu().squeeze(0).numpy().transpose((1, 2, 0))
    rect_two = grad_cam_output[1][2].cpu().squeeze(0).numpy().transpose((1, 2, 0))

    # Convert the NumPy arrays to PIL Images
    image_one_pil = Image.fromarray((image_one * 255).astype('uint8'))
    image_two_pil = Image.fromarray((image_two * 255).astype('uint8'))
    image_three_pil = Image.fromarray((image_three * 255).astype('uint8'))
    heatmap_pil = Image.fromarray((heatmap * 255).astype('uint8'))
    rect_one_pil = Image.fromarray((rect_one * 255).astype('uint8'))
    rect_two_pil = Image.fromarray((rect_two * 255).astype('uint8'))

    # Create the PhotoImage objects
    image_one = ImageTk.PhotoImage(image_one_pil)
    image_two = ImageTk.PhotoImage(image_two_pil)
    image_three = ImageTk.PhotoImage(image_three_pil)
    heatmap = ImageTk.PhotoImage(heatmap_pil)
    rect_one = ImageTk.PhotoImage(rect_one_pil)
    rect_two = ImageTk.PhotoImage(rect_two_pil)

    # Update the first panel
    panel1.config(image=image_one)
    panel1.image = image_one

    # Update the second panel
    panel2.config(image=image_two)
    panel2.image = image_two

    # Update the third panel
    panel3.config(image=image_three)
    panel3.image = image_three

    # Update the fourth panel
    panel4.config(image=heatmap)
    panel4.image = heatmap

    # Update the fifth panel
    panel5.config(image=rect_one)
    panel5.image = rect_one

    # Update the sixth panel
    panel6.config(image=rect_two)
    panel6.image = rect_two

    # Show the the texts once the prediction is done
    text_label_panel1.grid(row=3, column=0)
    text_label_panel2.grid(row=3, column=1)
    text_label_result.grid(row=5, column=0, columnspan=2)

    # Update the result label with the predicted class
    pred_text = decode_prediction(predicted_class)
    result_text.set(f"{pred_text[0]} ({'Benign' if pred_text[1] == 0 else 'Malignant'})")
    if pred_text[1] == 0:
        result_label.config(fg="green")
    else:
        result_label.config(fg="red")

def set_window_size():
    # Get the screen width and height
    #screen_width = root.winfo_screenwidth()
    #screen_height = root.winfo_screenheight()

    # Set the window size
    window_width = 480  # Adjust the width as needed
    window_height = 450  # Adjust the height as needed

    # Set the window size (80% of the screen width and height)
    #window_width = int(0.3 * screen_width)
    #window_height = int(0.3 * screen_height)

    # Set the window geometry
    root.geometry(f"{window_width}x{window_height}+{int((root.winfo_screenwidth() - window_width) / 2)}+{int((root.winfo_screenheight() - window_height) / 2)}")
    
    # Disable window resizing
    root.resizable(False, False)

# Create the main window
root = tk.Tk()
root.title("Melanoma Detection Demo")

# Set window size based on desktop dimensions
set_window_size()

# Create a title label
title_label = tk.Label(root, text="Melanoma Detection", font=("Helvetica", 16, "underline"), bg="pink")
title_label.grid(row=0, column=0, columnspan=2, pady=10)

# Create a description
description_label = tk.Label(root, text="Please, upload an image with a mole to make the diagnosis:", font=("Helvetica", 12))
description_label.grid(row=1, column=0, columnspan=2, pady=10)

# Create a button to open an image
open_button = tk.Button(root, text="Upload Image", command=open_image)
open_button.grid(row=2, column=0, columnspan=2)

# Create two panels with a small border between them
panel1 = tk.Label(root)
panel2 = tk.Label(root)

# Create three panels with a small border between them
panel3 = tk.Label(root)
panel4 = tk.Label(root)
panel5 = tk.Label(root)
panel6 = tk.Label(root)

# Create a label for text above panel 1
text_label_panel1 = tk.Label(root, text="Uploaded image", font=("Helvetica", 12))
text_label_panel1.grid_remove()

# Create a label for text above panel 2
text_label_panel2 = tk.Label(root, text="Segmented mole", font=("Helvetica", 12))
text_label_panel2.grid_remove()

# Create a label for text above panel 3
text_label_panel3 = tk.Label(root, text="Segmented mole", font=("Helvetica", 12))
text_label_panel3.grid_remove()

# Create a label for text above panel 4
text_label_panel4 = tk.Label(root, text="Heatmap", font=("Helvetica", 12))
text_label_panel4.grid_remove()

# Create a label for text above panel 5
text_label_panel5 = tk.Label(root, text="Bounding box", font=("Helvetica", 12))

# Create a label for text above panel 6
text_label_panel6 = tk.Label(root, text="Bounding box", font=("Helvetica", 12))

# Center panels in the window
panel1.grid(row=4, column=0, padx=5)
panel2.grid(row=4, column=1, padx=5)
panel3.grid(row=4, column=0, padx=5)
panel4.grid(row=4, column=1, padx=5)
panel5.grid(row=4, column=0, padx=5)
panel6.grid(row=4, column=1, padx=5)

# Create a label to display the result
text_label_result = tk.Label(root, text="Result of the diagnosis:", font=("Helvetica", 12, "underline"))
text_label_result.grid_remove()
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Helvetica", 12))
result_label.grid(row=6, column=0, columnspan=2)

# Assign the panels a mock image
grey_image_array = np.ones((224, 224, 3), dtype=np.uint8) * (240, 240, 240)
grey_image_pil = Image.fromarray(grey_image_array.astype('uint8'))
grey_image = ImageTk.PhotoImage(grey_image_pil)
panel1.config(image=grey_image)
panel1.image = grey_image
panel2.config(image=grey_image)
panel2.image = grey_image

# Start the GUI event loop
root.mainloop()