import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import time
import re

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from deepface import DeepFace
from deepface.extendedmodels import Age
from deepface.commons import functions, realtime, distance as dst
from deepface.detectors import OpenCvWrapper


def frame_level_facial_analysis(enable_face_analysis = True, source = 0, time_threshold = 5, frame_threshold = 5):

	input_shape = (224, 224); input_shape_x = input_shape[0]; input_shape_y = input_shape[1]

	text_color = (255,255,255)

	if enable_face_analysis == True:

		tic = time.time()

		emotion_model = DeepFace.build_model('Emotion')
		print("Emotion model loaded")

		#age_model = DeepFace.build_model('Age')

		#gender_model = DeepFace.build_model('Gender')

		toc = time.time()

		print("Facial attibute analysis models loaded in ",toc-tic," seconds")

	#-----------------------

	pivot_img_size = 112 #face recognition result image

	#-----------------------

	opencv_path = OpenCvWrapper.get_opencv_path()
	face_detector_path = opencv_path+"haarcascade_frontalface_default.xml"
	face_cascade = cv2.CascadeClassifier(face_detector_path)

	#-----------------------

	freeze = False
	face_detected = False
	face_included_frames = 0 #freeze screen if face detected sequantially 5 frames
	freezed_frame = 0
	tic = time.time()

	cap = cv2.VideoCapture(source) #webcam

	while(True):
		ret, img = cap.read()

		if img is None:
			break

		#cv2.namedWindow('img', cv2.WINDOW_FREERATIO)
		#cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

		raw_img = img.copy()
		resolution = img.shape

		resolution_x = img.shape[1]; resolution_y = img.shape[0]

		if freeze == False:
			faces = face_cascade.detectMultiScale(img, 1.3, 5)

			if len(faces) == 0:
				face_included_frames = 0
		else:
			faces = []

		detected_faces = []
		face_index = 0
		for (x,y,w,h) in faces:
			if w > 130: #discard small detected faces

				face_detected = True
				if face_index == 0:
					face_included_frames = face_included_frames + 1 #increase frame for a single face

				cv2.rectangle(img, (x,y), (x+w,y+h), (67,67,67), 1) #draw rectangle to main image

				cv2.putText(img, str(frame_threshold - face_included_frames), (int(x+w/4),int(y+h/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)

				detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face

				#-------------------------------------

				detected_faces.append((x,y,w,h))
				face_index = face_index + 1

				#-------------------------------------

		if face_detected == True and face_included_frames == frame_threshold and freeze == False:
			freeze = True
			#base_img = img.copy()
			base_img = raw_img.copy()
			detected_faces_final = detected_faces.copy()
			tic = time.time()

		if freeze == True:

			toc = time.time()
			if (toc - tic) < time_threshold:

				if freezed_frame == 0:
					freeze_img = base_img.copy()
					#freeze_img = np.zeros(resolution, np.uint8) #here, np.uint8 handles showing white area issue

					for detected_face in detected_faces_final:
						x = detected_face[0]; y = detected_face[1]
						w = detected_face[2]; h = detected_face[3]

						cv2.rectangle(freeze_img, (x,y), (x+w,y+h), (67,67,67), 1) #draw rectangle to main image

						#-------------------------------

						#apply deep learning for custom_face

						custom_face = base_img[y:y+h, x:x+w]

						#-------------------------------
						#facial attribute analysis

						if enable_face_analysis == True:

							gray_img = functions.preprocess_face(img = custom_face, target_size = (48, 48), grayscale = True, enforce_detection = False)
							emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
							emotion_predictions = emotion_model.predict(gray_img)[0,:]
							sum_of_predictions = emotion_predictions.sum()

							mood_items = []
							for i in range(0, len(emotion_labels)):
								mood_item = []
								emotion_label = emotion_labels[i]
								emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
								mood_item.append(emotion_label)
								mood_item.append(emotion_prediction)
								mood_items.append(mood_item)

							emotion_df = pd.DataFrame(mood_items, columns = ["emotion", "score"])
							emotion_df = emotion_df.sort_values(by = ["score"], ascending=False).reset_index(drop=True)

							#background of mood box

							#transparency
							overlay = freeze_img.copy()
							opacity = 0.4

							if x+w+pivot_img_size < resolution_x:
								#right
								cv2.rectangle(freeze_img
									#, (x+w,y+20)
									, (x+w,y)
									, (x+w+pivot_img_size, y+h)
									, (64,64,64),cv2.FILLED)

								cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

							elif x-pivot_img_size > 0:
								#left
								cv2.rectangle(freeze_img
									#, (x-pivot_img_size,y+20)
									, (x-pivot_img_size,y)
									, (x, y+h)
									, (64,64,64),cv2.FILLED)

								cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

							for index, instance in emotion_df.iterrows():
								emotion_label = "%s " % (instance['emotion'])
								emotion_score = instance['score']/100

								bar_x = 35 #this is the size if an emotion is 100%
								bar_x = int(bar_x * emotion_score)

								if x+w+pivot_img_size < resolution_x:

									text_location_y = y + 20 + (index+1) * 20
									text_location_x = x+w

									if text_location_y < y + h:
										cv2.putText(freeze_img, emotion_label, (text_location_x, text_location_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

										cv2.rectangle(freeze_img
											, (x+w+70, y + 13 + (index+1) * 20)
											, (x+w+70+bar_x, y + 13 + (index+1) * 20 + 5)
											, (255,255,255), cv2.FILLED)

								elif x-pivot_img_size > 0:

									text_location_y = y + 20 + (index+1) * 20
									text_location_x = x-pivot_img_size

									if text_location_y <= y+h:
										cv2.putText(freeze_img, emotion_label, (text_location_x, text_location_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

										cv2.rectangle(freeze_img
											, (x-pivot_img_size+70, y + 13 + (index+1) * 20)
											, (x-pivot_img_size+70+bar_x, y + 13 + (index+1) * 20 + 5)
											, (255,255,255), cv2.FILLED)

							#-------------------------------

							face_224 = functions.preprocess_face(img = custom_face, target_size = (224, 224), grayscale = False, enforce_detection = False)

						tic = time.time() #in this way, freezed image can show 5 seconds

						#-------------------------------


				cv2.imshow('img', freeze_img)

				freezed_frame = freezed_frame + 1
			else:
				face_detected = False
				face_included_frames = 0
				freeze = False
				freezed_frame = 0

		else:
			cv2.imshow('img',img)

		if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
			break

	#kill open cv things
	cap.release()
	cv2.destroyAllWindows()



def facial_analysis(enable_face_analysis = True, source = 0, time_threshold = 5, frame_threshold = 5):

	input_shape = (224, 224); input_shape_x = input_shape[0]; input_shape_y = input_shape[1]

	text_color = (255,255,255)

	if enable_face_analysis == True:

		tic = time.time()

		emotion_model = DeepFace.build_model('Emotion')
		print("Emotion model loaded")

		#age_model = DeepFace.build_model('Age')

		#gender_model = DeepFace.build_model('Gender')

		toc = time.time()

		print("Facial attibute analysis models loaded in ",toc-tic," seconds")

	#-----------------------

	pivot_img_size = 112 #face recognition result image

	#-----------------------

	opencv_path = OpenCvWrapper.get_opencv_path()
	face_detector_path = opencv_path+"haarcascade_frontalface_default.xml"
	face_cascade = cv2.CascadeClassifier(face_detector_path)

	#-----------------------

	freeze = False
	face_detected = False
	face_included_frames = 0 #freeze screen if face detected sequantially 5 frames
	freezed_frame = 0
	tic = time.time()

	cap = cv2.VideoCapture(source) #webcam

	while(True):
		ret, img = cap.read()

		if img is None:
			break

		#cv2.namedWindow('img', cv2.WINDOW_FREERATIO)
		#cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

		raw_img = img.copy()
		resolution = img.shape

		resolution_x = img.shape[1]; resolution_y = img.shape[0]

		if freeze == False:
			faces = face_cascade.detectMultiScale(img, 1.3, 5)

			if len(faces) == 0:
				face_included_frames = 0
		else:
			faces = []

		detected_faces = []
		face_index = 0
		for (x,y,w,h) in faces:
			if w > 130: #discard small detected faces

				face_detected = True
				if face_index == 0:
					face_included_frames = face_included_frames + 1 #increase frame for a single face

				cv2.rectangle(img, (x,y), (x+w,y+h), (67,67,67), 1) #draw rectangle to main image

				cv2.putText(img, str(frame_threshold - face_included_frames), (int(x+w/4),int(y+h/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)

				detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face

				#-------------------------------------

				detected_faces.append((x,y,w,h))
				face_index = face_index + 1

				#-------------------------------------

		if face_detected == True and face_included_frames == frame_threshold and freeze == False:
			freeze = True
			#base_img = img.copy()
			base_img = raw_img.copy()
			detected_faces_final = detected_faces.copy()
			tic = time.time()

		if freeze == True:

			toc = time.time()
			if (toc - tic) < time_threshold:

				if freezed_frame == 0:
					freeze_img = base_img.copy()
					#freeze_img = np.zeros(resolution, np.uint8) #here, np.uint8 handles showing white area issue

					for detected_face in detected_faces_final:
						x = detected_face[0]; y = detected_face[1]
						w = detected_face[2]; h = detected_face[3]

						cv2.rectangle(freeze_img, (x,y), (x+w,y+h), (67,67,67), 1) #draw rectangle to main image

						#-------------------------------

						#apply deep learning for custom_face

						custom_face = base_img[y:y+h, x:x+w]

						#-------------------------------
						#facial attribute analysis

						if enable_face_analysis == True:

							gray_img = functions.preprocess_face(img = custom_face, target_size = (48, 48), grayscale = True, enforce_detection = False)
							emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
							emotion_predictions = emotion_model.predict(gray_img)[0,:]
							sum_of_predictions = emotion_predictions.sum()

							mood_items = []
							for i in range(0, len(emotion_labels)):
								mood_item = []
								emotion_label = emotion_labels[i]
								emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
								mood_item.append(emotion_label)
								mood_item.append(emotion_prediction)
								mood_items.append(mood_item)

							emotion_df = pd.DataFrame(mood_items, columns = ["emotion", "score"])
							emotion_df = emotion_df.sort_values(by = ["score"], ascending=False).reset_index(drop=True)

							#background of mood box

							#transparency
							overlay = freeze_img.copy()
							opacity = 0.4

							if x+w+pivot_img_size < resolution_x:
								#right
								cv2.rectangle(freeze_img
									#, (x+w,y+20)
									, (x+w,y)
									, (x+w+pivot_img_size, y+h)
									, (64,64,64),cv2.FILLED)

								cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

							elif x-pivot_img_size > 0:
								#left
								cv2.rectangle(freeze_img
									#, (x-pivot_img_size,y+20)
									, (x-pivot_img_size,y)
									, (x, y+h)
									, (64,64,64),cv2.FILLED)

								cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

							for index, instance in emotion_df.iterrows():
								emotion_label = "%s " % (instance['emotion'])
								emotion_score = instance['score']/100

								bar_x = 35 #this is the size if an emotion is 100%
								bar_x = int(bar_x * emotion_score)

								if x+w+pivot_img_size < resolution_x:

									text_location_y = y + 20 + (index+1) * 20
									text_location_x = x+w

									if text_location_y < y + h:
										cv2.putText(freeze_img, emotion_label, (text_location_x, text_location_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

										cv2.rectangle(freeze_img
											, (x+w+70, y + 13 + (index+1) * 20)
											, (x+w+70+bar_x, y + 13 + (index+1) * 20 + 5)
											, (255,255,255), cv2.FILLED)

								elif x-pivot_img_size > 0:

									text_location_y = y + 20 + (index+1) * 20
									text_location_x = x-pivot_img_size

									if text_location_y <= y+h:
										cv2.putText(freeze_img, emotion_label, (text_location_x, text_location_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

										cv2.rectangle(freeze_img
											, (x-pivot_img_size+70, y + 13 + (index+1) * 20)
											, (x-pivot_img_size+70+bar_x, y + 13 + (index+1) * 20 + 5)
											, (255,255,255), cv2.FILLED)

							#-------------------------------

							face_224 = functions.preprocess_face(img = custom_face, target_size = (224, 224), grayscale = False, enforce_detection = False)

						tic = time.time() #in this way, freezed image can show 5 seconds

						#-------------------------------


				cv2.imshow('img', freeze_img)

				freezed_frame = freezed_frame + 1
			else:
				face_detected = False
				face_included_frames = 0
				freeze = False
				freezed_frame = 0

		else:
			cv2.imshow('img',img)

		if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
			break

	#kill open cv things
	cap.release()
	cv2.destroyAllWindows()



