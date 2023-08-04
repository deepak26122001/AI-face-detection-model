import cv2

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")
name_list = ["", "Deepak Cutie","SHAKIRA","BEYONCE","PRINCESS"]

while True:
    ret, frame=video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
       serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
       if conf > 10:
           cv2.putText(frame, name_list[serial],(x,y-40),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,255), 2)
           cv2.rectangle(frame, (x,y), (x+w,y+h), (50,50,255), 1) 
       else:
           cv2.putText(frame, "Not Recognized",(x,y-40),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,255), 2)
           cv2.rectangle(frame, (x,y), (x+w,y+h), (50,50,255), 1)
    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)

    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
print("Dateset collection mission achieved")

