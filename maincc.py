#Liberiras
import cv2
import mediapipe as mp
import numpy as np
mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose


#Streamlit 

import streamlit as st
import pandas as pd
header = st.container()

choice=0
count = 0
#detector = mp.poseDetector()

with header:
    st.title('Entrenador Personal Vision Artificial')
    st.header('Seleccione el ejercicio')
    

# Botón "Bicep"
if st.button("Doble Curl"):
    choice =int(1)
    dbg = "doubleb.gif"
    st.image(dbg, use_column_width=True)

# Botón "Hombro"
if st.button("Press Militar"):
    choice =int(2)
    pressm = "pressm.gif"
    st.image(pressm, use_column_width=True)

# Botón "Elevacion R"
if st.button("Elevacion de rodilla"):
    choice =int(3)
    rodillase = "rodillase.gif"
    st.image(rodillase, use_column_width=True)
    
# Botón "Elevacion R"
if st.button("Sentadilla"):
    choice =int(4)
    sentadilla = "sentadilla.gif"
    st.image(sentadilla, use_column_width=True)
# Botón "Abdominales"
if st.button("Abdominales"):
    choice =int(5)
    abs = "abs.gif"
    st.image(abs, use_column_width=True)

#with user_input:
    #st.header('Ingresa los datos')
    #fn1 = st.text_input("Enter your name ")
#choice =int(st.number_input('Ejercicio: 1.Bicep curl 2.Press Militar 4.-Lagartijas',max_value=4, step=1))
  


#calcular ángulos 


def calculate_angle(a,b,c):
    a = np.array(a)  #Primero Angulo
    b = np.array(b)  #Segundo Angulo
    c = np.array(c)  #Tercero Angulo
    
    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle>180.0:
        angle = 360-angle
    return angle


#Curl Contador 


cap = cv2.VideoCapture(0)

#Variable contadora del curl
counter = 0
stage = None
error=0

## Incializa media pepipe
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Cambiar el color de la imagen a RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Crea una decision 
        results = pose.process(image)
    
        # Volver a colorear a BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #Seleccion de ejercicio
        
        
        
        
        if(choice==1):
            try:
                 # extrae puntos de referencia
                landmarks = results.pose_landmarks.landmark
                
                #  Coordenadas que se ocupan
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Calcula angulo
                angle_1 =calculate_angle(left_shoulder, left_elbow, left_wrist)
                angle_2=calculate_angle(right_shoulder,right_elbow,right_wrist)

                # Visualiza el angulo
                cv2.putText(image, str(angle_1), 
                               tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )

                cv2.putText(image, str(angle_2), 
                               tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                

                #Cuenta las reps
                if angle_1 > 160 and angle_2 >160 :
                    stage = "Abajo"
                if angle_1 < 30 and angle_2 < 30 and stage == 'Abajo':
                    stage = "Arriba"
                    counter +=1
                    print(counter)

            except:
                pass
            
            #Rectangulo donde se muestran las reps
            cv2.rectangle(image, (0,0), (360,72), (245,117,16), -1)
            
            #Mnanda los valores al rectangulo
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            #Imprime los mensajes

            cv2.putText(image, 'STAGE', (165,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (165,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            cv2.putText(image, 'DOUBLE BICEP CURLS', (390,18), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            if counter >=1 and counter <2:
                        cv2.putText(image, '1 Set completo ', (390,90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            if counter >=2 and counter <3:
                        cv2.putText(image, '2 Set completo ', (390,90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                        
            if counter >=3 and counter <4:
                        cv2.putText(image, '3 Set completo ', (390,90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                        
            if counter >=4 and counter <5:
                        cv2.putText(image, '4 Set completo ', (390,90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                        
            if counter >=5 and counter <6:
                        cv2.putText(image, '5 Set completo ', (390,90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                        
            if counter >=6 and counter <7:
                        cv2.putText(image, '6 Set completo ', (390,90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                        
            if counter >=7 and counter <8:
                        cv2.putText(image, '7 Set completo ', (390,90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                                   
            if counter >=8 and counter <9:
                        cv2.putText(image, '8 Set completo ', (390,90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                        
            if counter >=9 and counter <10:
                        cv2.putText(image, '9 Set completo ', (390,90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                        
            if counter ==10 :
                        cv2.putText(image, '10 Set completo ', (390,90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            if counter >10 :
                        cv2.rectangle(image, (40,200), (600,72), (255,255,255), -1)


                        cv2.putText(image, 'Ejercicio Terminado', (60,150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 1, cv2.LINE_AA)

            # Procesa las detecciones
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                     )
            
        elif(choice==2):
                try:
                     # extrae puntos de referencia
                    landmarks = results.pose_landmarks.landmark

                    #  Coordenadas que se ocupan
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    # Calcula angulo
                    angle_1 = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    angle_2=calculate_angle(right_shoulder,right_elbow,right_wrist)

                    # Visualiza el angulo
                    cv2.putText(image, str(angle_1), 
                                   tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )

                    cv2.putText(image, str(angle_2), 
                                   tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )

                    #Cuenta las reps
                    if angle_1 > 110 and angle_2 >110 :
                        stage = "Arriba"
                    if angle_1 < 90 and angle_2 < 90 and stage == 'Arriba':
                        stage = "Abajo"
                        counter +=1
                        print(counter)


                except:
                    pass

                #Rectangulo donde se muestran las reps
                cv2.rectangle(image, (0,0), (360,72), (245,117,16), -1)

                #Manda los valores al rectangulo
                cv2.putText(image, 'REPS', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), 
                            (10,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                #Imprime los mensajes

                cv2.putText(image, 'STAGE', (165,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, 
                            (165,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                cv2.putText(image, 'Press Militar', (390,18), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                if counter >=1 and counter <2:
                            cv2.putText(image, '1 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                if counter >=2 and counter <3:
                            cv2.putText(image, '2 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                            
                if counter >=3 and counter <4:
                            cv2.putText(image, '3 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                if counter >=4 and counter <5:
                            cv2.putText(image, '4 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                if counter >=5 and counter <6:
                            cv2.putText(image, '5 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                if counter >=6 and counter <7:
                            cv2.putText(image, '6 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                if counter >=7 and counter <8:
                            cv2.putText(image, '7 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                if counter >=8 and counter <9:
                            cv2.putText(image, '8 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                if counter >=9 and counter <10:
                            cv2.putText(image, '9 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                if counter ==10 :
                            cv2.putText(image, '10 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                if counter >10 :

                            cv2.rectangle(image, (40,200), (600,72), (255,255,255), -1)


                            cv2.putText(image, 'Ejercicio Terminado', (60,150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 1, cv2.LINE_AA)

                # Procesa las detecciones
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                         )            
        
        elif(choice==3):
                
                # extrae puntos de referencia
                try:
                    landmarks = results.pose_landmarks.landmark
                    #  Coordenadas que se ocupan
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                    # Calcula angulo
                    angle = calculate_angle(shoulder, hip, knee)

                    # Visualiza el angulo
                    cv2.putText(image, str(angle), 
                                   tuple(np.multiply(hip, [640, 480]).astype(int)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )

                    #Cuenta las reps
                    if angle > 160:
                        stage = "Abajo"
                    if angle < 80 and stage == 'Abajo':
                        stage = "Arriba"
                        counter +=1
                        print(counter)


                except:
                    pass

               #Rectangulo donde se muestran las reps
                cv2.rectangle(image, (0,0), (360,72), (245,117,16), -1)

                #Manda los valores al rectangulo
                cv2.putText(image, 'REPS', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), 
                            (10,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                #Imprime los mensajes

                cv2.putText(image, 'STAGE', (165,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, 
                            (165,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                cv2.putText(image, 'Elevacion Rodillas', (390,18), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                if counter >=1 and counter <2:
                            cv2.putText(image, '1 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                if counter >=2 and counter <3:
                            cv2.putText(image, '2 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                if counter >=3 and counter <4:
                            cv2.putText(image, '3 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                if counter >=4 and counter <5:
                            cv2.putText(image, '4 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                if counter >=5 and counter <6:
                            cv2.putText(image, '5 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                if counter >=6 and counter <7:
                            cv2.putText(image, '6 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                if counter >=7 and counter <8:
                            cv2.putText(image, '7 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                if counter >=8 and counter <9:
                            cv2.putText(image, '8 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                if counter >=9 and counter <10:
                            cv2.putText(image, '9 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                if counter ==10 :
                            cv2.putText(image, '10 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                if counter >10 :

                            cv2.rectangle(image, (40,200), (600,72), (255,255,255), -1)


                            cv2.putText(image, 'Ejercicio Terminado', (60,150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 1, cv2.LINE_AA)

                # Procesa las detecciones
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                         )
        elif(choice==4):
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Obtener las coordenadas de los landmarks
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    # Calcular ángulos
                    angle_1 = calculate_angle(left_hip, left_knee, left_ankle)
                    angle_2 = calculate_angle(right_hip, right_knee, right_ankle)

                    # Visualizar ángulos
                    cv2.putText(image, str(angle_1), tuple(np.multiply(left_knee, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, str(angle_2), tuple(np.multiply(right_knee, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                    # Contador de sentadillas
                    if angle_1 > 160 and angle_2 > 160:
                        stage = "Abajo"
                    if angle_1 < 100 and angle_2 < 100 and stage == 'Abajo':
                        stage = "Arriba"
                        counter += 1

                    

                except:
                    pass
                
                #Rectangulo donde se muestran las reps
                cv2.rectangle(image, (0,0), (360,72), (245,117,16), -1)

                #Manda los valores al rectangulo
                cv2.putText(image, 'REPS', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), 
                            (10,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                #Imprime los mensajes

                cv2.putText(image, 'STAGE', (165,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, 
                            (165,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                cv2.putText(image, 'Sentadilla', (390,18), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                if counter >=1 and counter <2:
                            cv2.putText(image, '1 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                if counter >=2 and counter <3:
                            cv2.putText(image, '2 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                if counter >=3 and counter <4:
                            cv2.putText(image, '3 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                if counter >=4 and counter <5:
                            cv2.putText(image, '4 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                if counter >=5 and counter <6:
                            cv2.putText(image, '5 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                if counter >=6 and counter <7:
                            cv2.putText(image, '6 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                if counter >=7 and counter <8:
                            cv2.putText(image, '7 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                if counter >=8 and counter <9:
                            cv2.putText(image, '8 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                if counter >=9 and counter <10:
                            cv2.putText(image, '10 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                if counter ==10 :
                            cv2.putText(image, '10 Set completo ', (390,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                if counter >10 :

                            cv2.rectangle(image, (40,200), (600,72), (255,255,255), -1)


                            cv2.putText(image, 'Ejercicio Completo', (60,150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 1, cv2.LINE_AA)
                # Procesa las detecciones
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                         )
        elif(choice==5):
                try:
                    # extrae puntos de referencia
                    landmarks = results.pose_landmarks.landmark
                    #  Coordenadas que se ocupan
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                    # Calcula angulo
                    angle = calculate_angle(shoulder, hip, knee)

                    # Visualiza el angulo
                    cv2.putText(image, str(angle), tuple(np.multiply(hip, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                    #Cuenta las reps
                    if angle > 160:
                        stage = "Abajo"
                    if angle < 80 and stage == 'Abajo':
                        stage = "Arriba"
                        counter += 1
                        print(counter)

                except:
                    pass

                #Rectangulo donde se muestran las reps
                cv2.rectangle(image, (0, 0), (360, 72), (245, 117, 16), -1)

                #Manda los valores al rectangulo
                cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                #Imprime los mensajes
                cv2.putText(image, 'STAGE', (165, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, (165, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, 'Abdominales', (390, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                if counter >= 1 and counter < 2:
                    cv2.putText(image, '1 Set completo ', (390, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                if counter >= 2 and counter < 3:
                     cv2.putText(image, '2 Set completo ', (390, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                if counter >= 3 and counter < 4:
                     cv2.putText(image, '3 Set completo ', (390, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  
                if counter >= 4 and counter < 5:
                     cv2.putText(image, '4 Set completo ', (390, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                if counter >= 5 and counter < 6:
                     cv2.putText(image, '5 Set completo ', (390, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)   
                if counter >= 6 and counter < 7:
                     cv2.putText(image, '6 Set completo ', (390, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                if counter >= 7 and counter < 8:
                     cv2.putText(image, '7 Set completo ', (390, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                if counter >= 8 and counter < 9:
                     cv2.putText(image, '8 Set completo ', (390, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                if counter >= 9 and counter < 10:
                     cv2.putText(image, '9 Set completo ', (390, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)   
                

                if counter == 10:
                    cv2.putText(image, '10 Set completo ', (390, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                if counter > 10:
                    cv2.rectangle(image, (40, 200), (600, 72), (255, 255, 255), -1)
                    cv2.putText(image, 'Ejercicio Terminado', (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1, cv2.LINE_AA)

                # Procesa las detecciones
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        else:
            print("Numero Incorrecto")
            break
        
        cv2.imshow('VIDEO', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    

    cap.release()
    cv2.destroyAllWindows()