import cv2
import numpy as np
from keras.models import load_model
import csv
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

def load_emotion_model(model_path='model_file.h5'):
	return load_model(model_path)

def preprocess_image(face_img):
	resized = cv2.resize(face_img, (48, 48))
	normalized = resized / 255.0
	reshaped = np.reshape(normalized, (1, 48, 48, 1))
	return reshaped

def detect_emotion(frame, face_cascade, emotion_model, labels_dict):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 3)
	for x, y, w, h in faces:
		sub_face_img = gray[y:y+h, x:x+w]
		processed_img = preprocess_image(sub_face_img)
		result = emotion_model.predict(processed_img)
		label = np.argmax(result, axis=1)[0]
		draw_rectangle_and_text(frame, x, y, w, h, labels_dict[label])
		

def draw_rectangle_and_text(frame, x, y, w, h, label):
	#frame[:] = (128, 0, 128)  # Purple color in BGR format
	cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)	
	cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 0, 128), 2)
	cv2.rectangle(frame, (x, y-40), (x+w, y), (128, 0, 128), -1)
	cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


	
def main(image_path, output_path="output_disgust.jpg"):
	model = load_emotion_model()
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
	'''emotion_counts = {emotion: 0 for emotion in labels_dict.values()}'''

	frame = cv2.imread(image_path)
	detect_emotion(frame, face_cascade, model, labels_dict)
	
	cv2.imwrite(output_path, frame)
	print(f"Output image saved to: {output_path}")

	cv2.imshow("Frame", frame)
	'''cv2.waitKey(0)'''
	cv2.destroyAllWindows()

	'''print("Emotion Count for the Single Image:")
	for emotion, count in emotion_counts.items():
		print(f"{emotion}: {count}")'''
	# Print confusion matrix
	cm = confusion_matrix(true_labels, predicted_labels, labels=list(labels_dict.keys()))
	print("Confusion Matrix:")
	print(cm)

	# Display confusion matrix as a plot
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(labels_dict.values()))
	disp.plot(cmap='viridis', values_format='d', xticks_rotation='vertical')
	plt.show()

if __name__ == "__main__":
    image_path = "disgust.jpg"  # Replace with the path to your image
    output_path = "output_disgust.jpg"  # Specify the desired output path
    main(image_path, output_path)
    

