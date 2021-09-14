import cv2
import ctypes
import numpy as np

cap = cv2.VideoCapture(0)

# 720p Recording

cap.set(5, 30)
cap.set(3, 1280)
cap.set(4, 720)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

print("[INFO] loading AI model...")
ageModel="models/age_net.caffemodel"
genderModel="models/gender_net.caffemodel"
#facemaskModel="models/facemask_net.caffemodel"
faceModel="models/face_net.caffemodel"

ageProto="proto/age_deploy.prototxt"
genderProto="proto/gender_deploy.prototxt"
#facemaskProto="proto/facemask_deploy.prototxt"
faceProto="proto/face_deploy.prototxt"

ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)
#facemaskNet=cv2.dnn.readNet(facemaskModel,facemaskProto)
faceNet=cv2.dnn.readNet(faceModel,faceProto)

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']
hasMaskList=['yes', 'no']

ctypes.windll.user32.MessageBoxW(0, "Data seen via webcam may be stored into your computer and will only be used for educational use only. The only data we track is Age, Gender, Time, Had A Mask. If you continue you agree to theses terms.", "Warning!", 1)

count = 0

while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))
    

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.putText(frame,'ShadowmaneAI',(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(155,155,155),2)
    cv2.putText(frame,f'Pepole in this room : ~{count}',(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)

    #Set Blob
    blob=cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    faceDetections = faceNet.forward()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # loop over the detections

    for i in range(0, faceDetections.shape[2]):
                # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = faceDetections[0, 0, i, 2]
        #print(confidence)

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.3:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = faceDetections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(width - 1, endX), min(height - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))

            cv2.rectangle(frame, (startX, startY), (endX, endY), (255,0,0), 2)

            # Gender 
            test_blob=cv2.dnn.blobFromImage(frame, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(test_blob)
            genderPreds=genderNet.forward()
            gender=genderList[genderPreds[0].argmax()]

            # Age
            ageNet.setInput(test_blob)
            agePreds=ageNet.forward()
            age=ageList[agePreds[0].argmax()]

            # Face Mask
            cv2.putText(frame,f'{gender} - {age}',(startX,endY+0),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            roi_gray = gray[startY:startY + height, startX:startX + width]
            roi_color = frame[startY:startY + height, startX:startX + width]
            eyes = eye_cascade.detectMultiScale(roi_gray,minNeighbors=15)
            for (ex, ey, ew, eh) in eyes:
                count = len(eyes)
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)



    #faces = face_cascade.detectMultiScale(gray, 1.3, 2)
    #for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #roi_gray = gray[y:y+w, x:x+w]
        #roi_color = frame[y:y+h, x:x+w]
        #face_img = frame[y:y+h, x:x+w]
        #blob=cv2.dnn.blobFromImage(face_img, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        
        # Gender 
        #genderNet.setInput(blob)
        #genderPreds=genderNet.forward()
        #gender=genderList[genderPreds[0].argmax()]
        #print(f'Gender: {gender}')

        # Age
        #ageNet.setInput(blob)
        #agePreds=ageNet.forward()
        #age=ageList[agePreds[0].argmax()]
        #print(f'Age: {age[1:-1]} years')

        # Facemask
        #facemaskNet.setInput(blob)
        #facemaskPreds=facemaskNet.forward()
        
        #cv2.putText(frame,f'NO MASK {facemask}',((x+w)//2,y+h+35),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

        #Overlay Data AGE + GENDER
        #cv2.putText(frame,f'{gender}, {age}',((x+w)//2,y+h-200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        #eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        #for (ex, ey, ew, eh) in eyes:
            #cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)


    cv2.imshow('ShadowmaneAI - Created by Lunar Lynix', frame)

    if cv2.waitKey(1) == ord('q'):
        break   

cap.release()
cv2.destroyAllWindows()