import cv2
import pyttsx3
import pytesseract
import os

#tesseract path
pytesseract.pytesseract.tesseract_cmd = r""C:\Users\Arun JH\Downloads\tesseract-ocr-w64-setup-5.5.0.20241111.exe""

#initialize text-to-speech engine 
engine = pyttsx3.init()


#Load class Labels
classNames = [
     "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

net = cv2.dnn.readNetFromCaffe(
     "models/MobileNetSSD_deploy.prototxt.txt",
    "models/MobileNetSSD_deploy.caffemodel"

)
 
 # Open the default camera
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error:could not open the webcam")
    exit()

# To avoid speaking of same object repeatedly
spoken_labels = set()    

while True:
    ret, frame = cam.read()    #To capture frame by frame 

    if not ret:
        break

    h, w = frame.shape[:2]

     # Preprocess input and 00.007834 normalizes the pixel value
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)   # preprocesses the image(frame) and convert into blob
    net.setInput(blob)                                                  # blob is the input format expected by the DNN
    detections = net.forward()

        # Loop over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:                  # higher threshold to reduce the noise
            idx = int(detections[0, 0, i, 1])
            label = classNames[idx]

            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # speak of new objects 
            if label not in spoken_labels:
                engine.say(label)
                engine.runAndWait()
                spoken_labels.add(label)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)

    if text.strip():  # check if text is not empty 
        print("Text Detected:",text.stip())
        engine.say("reading text:", text.stip())
        engine.runAndWait()

    cv2.imshow("AI reader:",frame)    
    if cv2.waitKey(1) & 0xFF == ord('q'):       #Pressing 'q' to quit and ord() will convert char 'q' into its ASCII value
        break

cam.release()
cv2.destroyAllWindows()
