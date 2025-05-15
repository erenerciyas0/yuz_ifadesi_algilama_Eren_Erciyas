#@markdown We implemented some functions to visualize the face landmark detection results. <br/> Run the following cell to activate the functions.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from PIL import Image, ImageDraw, ImageFont

# Modeli yükle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Türkçe + emoji açıklamaları
etiket_aciklama = {
    "happy": ("Mutlu", "😄"),
    "sad": ("Üzgün", "😢"),
    "angry": ("Kızgın", "😠"),
    "surprised": ("Şaşkın", "😲")
}

etiket_renkleri = {
    "happy": (0, 255, 0),
    "sad": (255, 0, 0),
    "angry": (0, 0, 255),
    "surprised": (255, 255, 0)
}

# Font dosyası yolları (Windows için)
font_yazi = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 50)       # Türkçe karakter desteği
font_emoji = ImageFont.truetype("C:/Windows/Fonts/seguiemj.ttf", 48)     # Emoji desteği

# Tahmini ekrana yazdıran fonksiyon
def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        koordinatlar = []
        for landmark in face_landmarks:
            koordinatlar.append(round(landmark.x, 4))
            koordinatlar.append(round(landmark.y, 4))

        sonuc = model.predict([koordinatlar])[0]
        ifade, emoji = etiket_aciklama.get(sonuc, (sonuc, ""))
        renk = etiket_renkleri.get(sonuc, (255, 255, 255))

        # OpenCV görüntüsünü PIL formatına çevir
        pil_image = Image.fromarray(annotated_image)
        draw = ImageDraw.Draw(pil_image)

        # Türkçe yazı ve emoji yaz
        draw.text((50, 50), ifade, font=font_yazi, fill=renk)
        draw.text((50, 110), emoji, font=font_emoji, fill=renk)

        # Geri OpenCV formatına çevir
        annotated_image = np.array(pil_image)

    return annotated_image

# Blendshape çizimi (opsiyonel)
def plot_face_blendshapes_bar_graph(face_blendshapes):
    face_blendshapes_names = [b.category_name for b in face_blendshapes]
    face_blendshapes_scores = [b.score for b in face_blendshapes]
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores)
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()

# Mediapipe model kurulumu
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Kamera başlatılır
cam = cv2.VideoCapture(0)
while cam.isOpened():
    basari, frame = cam.read()
    if basari:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        detection_result = detector.detect(mp_image)

        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
        cv2.imshow("Yüz İfadesi Algılama", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            break

cam.release()
cv2.destroyAllWindows()
