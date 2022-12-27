import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# 좌측 눈, 우측 눈의 좌표
leftEyeIris = [468, 469, 470, 471, 472]
rightEyeIris = [473, 474, 475, 476, 477]

def _normalized_to_pixel_coordinates(
    normalized_x, normalized_y, image_width,
    image_height):
  """
  Converts normalized value pair to pixel coordinates.
  이미지 크기에 따라서 상대적 좌표의 x, y를 반환하는 함수
  """
  space = 3

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    return None

  x_px = min(round(normalized_x * image_width, space), image_width - 1)
  y_px = min(round(normalized_y * image_height, space), image_height - 1)
  return x_px, y_px

def _draw_dot_in_coordination(
  eye_iris: list[int],
  label: str,
  color: tuple[int],
  position: tuple[int]
):
  """
  draw dot with coordination.
  주어진 좌표를 기반으로 점을 찍는 함수
  """
  # 0번째가 눈의 중앙
  for pos in eye_iris:
    mark = results.multi_face_landmarks[0].landmark[pos]
    # mark.x, mark.y, mark.z을 통하여 해당 좌표의 (x, y, z)를 활용할 수 있습니다.
    
    keypoint_px = _normalized_to_pixel_coordinates(mark.x, mark.y,
                                                   image_cols, image_rows)
    cv2.circle(image, (math.floor(keypoint_px[0]), math.floor(keypoint_px[1])), 1, color, 1)
    if pos == eye_iris[0]:
      cv2.putText(image, str(f'{label} ({keypoint_px[0]:.3f}, {keypoint_px[1]:.3f})'), position, 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA) 

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()

    if not success:
      continue

    image.flags.writeable = False
    image_rows, image_cols, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks: # if detected face.
      try:
        _draw_dot_in_coordination(leftEyeIris, 'Left eye', (0, 255, 0), (30, 30))
        _draw_dot_in_coordination(rightEyeIris, 'Right eye', (0, 0, 255), (30, 70))
      except Exception as err:
        print(err)

    cv2.imshow('Honeymind Iris', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
