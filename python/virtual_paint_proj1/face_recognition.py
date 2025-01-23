import cv2 as cv
#for handling the files
import os
#used for the matrix m
import numpy as np

#Directory person folder are save
DIR=r"C:\Users\kkamb\OneDrive\Desktop\code_learn\python\opencv\faces"
#load the harscascade file
cascade_image=cv.CascadeClassifier(r'C:\Users\kkamb\OneDrive\Desktop\code_learn\python\opencv\haarcascade_frontalface_default.xml')

#this function is used to capture the image
def capture_image(person_name):
    #person directory
    path_person=os.path.join(DIR,person_name)
    os.mkdir(path_person)
    #capture the image
    cap = cv.VideoCapture(0)
    count=0

    while True:
        success, frame=cap.read()
        cv.imshow('Image',frame)

       
        
        #convert into the grey
        gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        #To detect the face
        faces=cascade_image.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3,minSize=(30,30))

       #crop the faces
        for (x,y,w,h) in faces:
            face_ro=gray[y:y+h,x:x+w]


            #save the path in peson directroy
            face_path=os.path.join(path_person,f"{person_name}.jpg")
            cv.imwrite(face_path, face_ro)
            count = count + 1

            #drawing a rectangle on the face
            rectangle=cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),thickness=3)

        cv.imshow('Detected face',frame)
        
        key=cv.waitKey(1)
        #ord function take a single character of as input like q
        if key == ord('q'):
            break
        
    cap.release()
    cv.destroyAllWindows()          

    print(f"Captured {count} image of {person_name}. stored in{face_path}")   

#create a model
def create_model():
    features = []
    labels = []

    # Load existing people if available
    people = list(os.listdir(DIR)) if os.path.exists(DIR) else []
    
    #iterate over a folder inside the image to find person name and person
    for label, person in enumerate(people):
        person_path = os.path.join(DIR, person)
        if not os.path.isdir(person_path):
            continue
         #finding the image path
        for img in os.listdir(person_path):
            img_path = os.path.join(person_path, img)
            #convert to gray image
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

            
            #add features and label at list
            features.append(img)
            labels.append(label)

  

    # Convert to numpy arrays
    features = np.array(features, dtype="object")
    labels = np.array(labels)

    # Train the LBPH face recognizer
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    #give the features and labels to train a model
    face_recognizer.train(features, labels)

    # Save the face model
    face_recognizer.save("face_model.yml")
    #save the people
    np.save("people.npy", people)

#fuction for face recognizer
def face_recognize():
         #check if the face_model is in exists in the directory or not       
    if not os.path.exists("face_model.yml") or not os.path.exists("people.npy"):
        print("Model or people data not found.")
        return
     #make a instance of face
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    #read the face from the facce_model
    face_recognizer.read("face_model.yml")
    #load the people
    people = np.load("people.npy", allow_pickle=True)
     #capture the video of person
    video = cv.VideoCapture(0)
    
                

    while True:
            success, frame=video.read()
        #convert this image into gray image
            gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            #detect the face using detectMultiscale
            face=cascade_image.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3,minSize=(40,40))
        #crop the face for detect the person
            for (x, y, w, h) in face:
                    face_roi = gray[y:y + h, x:x + w]

                    #predict the confidence and label of the person
                
                    label, confidence = face_recognizer.predict(face_roi)
                    print(f'the name of the person is {people[label]} with confidence {confidence}')
                
                    if confidence < 70:
                         name = people[label]
                         color=(0,255,0)  #Green for the real faces
                 #resolution of the phone is 640*480 threshold value is 3000and 4000 then the face is on mobile and paper
                   #if the resolution of video 1920*1080 theresholod value above 8000 then the face is real
                    elif w * h < 8000:
                        color=(0,0,255)#Red for the mobile and paper faces
                    else:
                        name='Unknown'
                        color=(0,0,255)#red for the unknown faces
                   
                    #show the name of the person on the image
                    cv.putText(frame, f"{name}", (x, y), cv.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    #make a rectangle around the face of the person
                    #(x,y) is a top-left and bottom-right is (x+w,y+h)
                    cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            #show the face
            cv.imshow('Face Recognition',frame)

            key= cv.waitKey(1)
            if key == ord('q'):
                break
    #realese the camera
    video.release()
    #destroy all the window
    cv.destroyAllWindows()

    
# person_name=input(f"Enter the person name: ")
# print(person_name)
#     #call the function
# capture_image(person_name)
# create_model()
# face_recognize()

#This is the main method that calls the function by using choice
if __name__ == "__main__":
    #checks the directory is present or not
    os.makedirs(DIR, exist_ok=True)

    print("1: Capture Images")
    print("2: Train Model")
    print("3: Recognize Faces")
    ch = input("Enter your choice: ")

    if ch == "1":
        person_name = input("Enter the name of the person: ")
        #call the capture the image of person
        capture_image(person_name)
    elif ch == "2":
        #call the create model
        create_model()
    elif ch == "3":
        #call the face recognization
        face_recognize()
    else:
        print("Invalid character")







         
