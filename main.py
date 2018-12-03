import cv2, glob, dlib

age_list = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list = ['Male', 'Female']

detector = dlib.get_frontal_face_detector()

age_net = cv2.dnn.readNetFromCaffe(
          'models/deploy_age.prototxt', 
          'models/age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe(
          'models/deploy_gender.prototxt', 
          'models/gender_net.caffemodel')

img_list = glob.glob('img/*.jpg')

for img_path in img_list:
  img = cv2.imread(img_path)

  faces = detector(img)

  for face in faces:
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

    face_img = img[y1:y2, x1:x2].copy()

    blob = cv2.dnn.blobFromImage(face_img, scalefactor=1, size=(227, 227),
      mean=(78.4263377603, 87.7689143744, 114.895847746),
      swapRB=False, crop=False)

    # predict gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]

    # predict age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[age_preds[0].argmax()]

    # visualize
    cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,255), 2)
    overlay_text = '%s %s' % (gender, age)
    cv2.putText(img, overlay_text, org=(x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
      fontScale=1, color=(0,0,0), thickness=10)
    cv2.putText(img, overlay_text, org=(x1, y1),
      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)

  cv2.imshow('img', img)
  cv2.imwrite('result/%s' % img_path.split('/')[-1], img)

  key = cv2.waitKey(0) & 0xFF
  if key == ord('q'):
    break
