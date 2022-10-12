import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

# For static images:
IMAGE_FILES = []
with mp_objectron.Objectron(static_image_mode=True,
                            max_num_objects=5,
                            min_detection_confidence=0.1,
                            model_name='Cup') as objectron:
  for idx, file in enumerate(IMAGE_FILES):
    print(file)
    image = cv2.imread(file)
    # image = Image.open(file)
    image = np.array(image)
    print(type(image))

    # print(image)

    # Convert the BGR image to RGB and process it with MediaPipe Objectron.
    results = objectron.process(image)
    # for i in results.detected_objects[0].landmarks_3d.landmark:
    #   print(i.z)
    # Draw box landmarks.
    if not results.detected_objects:
      print(f'No box landmarks detected on {file}')
      continue
    print(f'Box landmarks of {file}:')
    annotated_image = image.copy()
    for detected_object in results.detected_objects:
      mp_drawing.draw_landmarks(
          annotated_image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)  #画框
      mp_drawing.draw_axis(annotated_image, detected_object.rotation,
                           detected_object.translation)  #画轴
      cv2.imwrite('C:\\Users\\BOB\\Downloads\\' + str(idx) + '.png', annotated_image)
      cv2.imshow('1',annotated_image)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
      # annotated_image = np.array(annotated_image)