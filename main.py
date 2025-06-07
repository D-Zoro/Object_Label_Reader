import cv2
 
 # Open the default camera
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error:could not open the webcam")
    exit()

while True:
    ret, frame = cam.read()    #To capture frame by frame 

    if not ret:
        print("Error: Failed to capture")
        break

    cv2.imshow("Webcam feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):       #Pressing 'q' to quit
        break

cam.release()
cv2.destroyAllWindows()
    