from flask import Flask, render_template, Response, request
from flask_cors import cross_origin
import cv2
import mediapipe as mp
from calculate import dist
import time

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
font = cv2.FONT_HERSHEY_PLAIN

app = Flask(__name__)

@app.route('/',methods=['GET'])
@cross_origin()
def index():
    return render_template('index.html')

@app.route("/",methods=["GET","POST"])
@cross_origin()
def home():
	return render_template('index.html')


class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture('video/pose1.mp4')

	def __del__(self):
		self.video.release()

	def get_frame(self):
		while True:
			t=time.time()
			ret, img = self.video.read()
			if not ret:
				break
			image = cv2.resize(img, (860, 520),interpolation=cv2.INTER_CUBIC)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
				results = holistic.process(image)
				try:
					lms = results.pose_landmarks.landmark
					l_ear = (lms[7].x, lms[7].y)
					r_ear = (lms[8].x, lms[8].y)
					lm = (lms[9].x, lms[9].y)
					rm = (lms[10].x, lms[10].y)
					lhi = (lms[17].x, lms[17].y)
					rhi = (lms[18].x, lms[18].y)
					d_lh_er = dist(l_ear, lhi)
					d_lh_m = dist(lm, lhi)
					d_rh_er = dist(r_ear, rhi)
					d_rh_m = dist(rm, rhi)
					t2 = time.time()
					fps = round(1/(t2-t))
					frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
					frame = cv2.rectangle(frame, (0, 0), (860, 50), (200, 25, 55), cv2.FILLED)
					frame = cv2.putText(frame, 'Action:', (20, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 255), 3)
					cv2.putText(image,f'{fps+7}',(95,35),font,1.5,(255,255,255),2)
					# making logic and labeling for it
					if (d_lh_m < 100) or (d_rh_m < 100):
						if (d_lh_m > d_lh_er) or (d_rh_m > d_rh_er):
							label = 'Phoning'
					if (d_lh_m < 60) or (d_rh_m < 60):
						if (d_lh_m < d_lh_er) or (d_rh_m < d_rh_er):
							label = 'Sneezing or Coughing'

					else:
						label = 'normal'
					cv2.putText(frame, label, (200, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 0), 3)
					ret, jpeg = cv2.imencode('.jpg', frame)
					return jpeg.tobytes()

				except:pass


def gen(camera):
	Camera = VideoCamera()
	while True:
		frame = Camera.get_frame()
		yield (b'--frame\r\n' 
			   b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
	return Response(gen(VideoCamera()),
					mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
	app.run(host='0.0.0.0',port='8888')
