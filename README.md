# WORKSHOP – 5  
# LICENSE PLATE DETECTION USING OPENCV AND HAAR CASCADE CLASSIFIER

---

# Aim

To detect objects from an image using OpenCV and Haar Cascade Classifier techniques in Python.

---

# Software Required

- Anaconda – Python 3.7  
- OpenCV  
- NumPy  
- Matplotlib  

---

# Algorithm

### Step 1:
Import the necessary packages.

### Step 2:
Read and display the input image.

### Step 3:
Convert the image into grayscale.

### Step 4:
Apply Gaussian Blur and Histogram Equalization for preprocessing.

### Step 5:
Load the Haar Cascade Classifier XML file.

### Step 6:
Detect objects using `detectMultiScale()`.

### Step 7:
Draw rectangles around detected objects.

### Step 8:
Save the detected object regions.

### Step 9:
Display the final detected output image.

---

# Program

### Developed By : Kiruba RC  
### Register Number : 212224230125

```python
import cv2
import matplotlib.pyplot as plt
import os
import urllib.request

# -------------------------------------------------------------
# Step 1: Read and display the input image
# -------------------------------------------------------------
image_path = 'img.jpg'

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(
        "Image not found. Please check the image_path variable."
    )

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')
plt.show()

# -------------------------------------------------------------
# Step 2: Convert to grayscale
# -------------------------------------------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()

# -------------------------------------------------------------
# Step 3: Preprocessing
# -------------------------------------------------------------
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

equalized = cv2.equalizeHist(blurred)

plt.imshow(equalized, cmap='gray')
plt.title("Preprocessed Image")
plt.axis('off')
plt.show()

# -------------------------------------------------------------
# Step 4: Load Haar Cascade
# -------------------------------------------------------------
cascade_path = 'haarcascade_frontalface_default.xml'

# Download cascade file automatically
if not os.path.exists(cascade_path):

    print("Cascade file not found. Downloading...")

    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

    urllib.request.urlretrieve(url, cascade_path)

    print("Cascade file downloaded successfully!")

# Load classifier
face_cascade = cv2.CascadeClassifier(cascade_path)

# -------------------------------------------------------------
# Step 5: Detect Faces
# -------------------------------------------------------------
faces = face_cascade.detectMultiScale(
    equalized,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

print(f"Total Faces Detected: {len(faces)}")

# -------------------------------------------------------------
# Step 6: Draw Rectangles and Save Faces
# -------------------------------------------------------------
output = image.copy()

save_dir = "Detected_Faces"

os.makedirs(save_dir, exist_ok=True)

for i, (x, y, w, h) in enumerate(faces):

    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        3
    )

    face_crop = image[y:y+h, x:x+w]

    save_path = f"{save_dir}/face_{i+1}.jpg"

    cv2.imwrite(save_path, face_crop)

if len(faces) > 0:
    print(f"{len(faces)} face(s) saved in '{save_dir}' folder.")
else:
    print("No faces detected.")

# -------------------------------------------------------------
# Step 7: Display Final Output
# -------------------------------------------------------------
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Detected Faces")
plt.axis('off')
plt.show()
```

---

# Output

## Original Image

<img width="406" height="519" alt="image" src="https://github.com/user-attachments/assets/afb22924-8aad-4d53-b86b-adba1957ad6c" />


## Grayscale Image

<img width="396" height="543" alt="image" src="https://github.com/user-attachments/assets/c16ce82d-923e-4b8e-b51f-c8a0202deb10" />


## Preprocessed Image

<img width="447" height="644" alt="image" src="https://github.com/user-attachments/assets/3dc619da-6f20-4b08-9079-090ebdee00e8" />


## Detected Output Image

<img width="426" height="510" alt="image" src="https://github.com/user-attachments/assets/f3c3e9de-431b-47bd-9d15-802f84ee7a5d" />


---

# Result

Thus the object detection using OpenCV and Haar Cascade Classifier was successfully implemented in Python, and the detected objects were displayed and saved successfully.
