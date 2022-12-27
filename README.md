## 눈동자 추적 프로그램

### 설치 및 실행 방법
```python
git clone https://github.com/HoneyMnB/eye-tracker.git .
pip install -r requirements.txt
python eye_tracker.py
```
### 파일 설명

|파일명|목적|
|------|---|
|eye_tracker.py|눈동자의 좌표에 관한 코드|

### 사용된 라이브러리
 - mediapipe Face Mesh Model : https://google.github.io/mediapipe/solutions/face_mesh
 - opencv-python


### 참고자료
 - [얼굴 좌표 Index 정보](https://github.com/google/mediapipe/blob/557cd050f3bf079266aaa7b88987a2cab5ab9ab3/mediapipe/python/solutions/face_mesh_connections.py#L1)