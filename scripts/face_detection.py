import cv2
import mediapipe as mp

# MediaPipe Face Detection 初期化
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# カメラ起動（0はデフォルトカメラ）
cap = cv2.VideoCapture(0)

# 顔検出器
with mp_face_detection.FaceDetection(
    model_selection=0,      # 0:近距離向け, 1:遠距離向け
    min_detection_confidence=0.5
) as face_detection:

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("カメラ画像を取得できません")
            break

        # BGR -> RGB変換
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 顔検出
        results = face_detection.process(rgb_frame)

        # 検出結果を描画
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)

        # 表示
        cv2.imshow("Face Detection", frame)

        # qキーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()