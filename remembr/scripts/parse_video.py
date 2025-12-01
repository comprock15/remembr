import cv2
import os
import argparse

def extract_frames(video_path, output_path):
    if not os.path.isfile(video_path):
        print(f"Ошибка: Файл {video_path} не найден!")
        return

    os.makedirs(output_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(fps))  

    if frame_interval == 0:
        print("Ошибка: Не удалось определить FPS видео!")
        return

    print(f"FPS видео: {fps:.2f}")
    print("Начало обработки...")

    saved_count = 0
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Сохраняем каждый n-ый кадр, где n = FPS
        if frame_number % frame_interval == 0:
            frame_name = f"frame_{saved_count:06d}.jpg"
            frame_path = os.path.join(output_path, frame_name)
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_number += 1

    cap.release()
    print(f"Готово! Сохранено кадров: {saved_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Извлечение кадров из видео')
    parser.add_argument('--input', required=True, help='Путь к исходному видеофайлу')
    parser.add_argument('--output', required=True, help='Путь для сохранения кадров')
    args = parser.parse_args()

    extract_frames(args.input, args.output)