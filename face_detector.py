import random
import threading
import time
import cv2
import pyttsx3
from deepface import DeepFace


# Инициализация синтеза речи
engine = pyttsx3.init()
engine.setProperty("rate", 140)  # Скорость речи
# Устанавливаем мужской голос, если есть поддержка мужского голоса,
# то поменяйте [0] на [3]
engine.setProperty("voice", engine.getProperty("voices")[0].id)

# Пути к фотографиям
wife_image_paths = [
    "images/img1.jpeg",
    "images/img2.jpeg",
    "images/img3.jpeg",
    "images/img4.jpeg",
    "images/img5.jpeg",
    "images/img6.jpeg"
]


# Таймер для предотвращения частого повторения приветствия
LAST_GREET_TIME = 0
GREETING_INTERVAL = 2  # Интервал между приветствиями (в секундах)

# Список возможных фраз
greetings = [
    "Привет дорогая! Как дела?",
    "Оо, это ты! Какая встреча!",
    "Рад тебя видеть! Как твои дела?",
    "Моя королева, добро пожаловать!",
    "Оо, ты выглядишь прекрасно сегодня!",
    "Свет моей жизни, ты вернулась!",
    "Я скучал! Как твой день прошел?",
    "Вот это удача! Любимая пришла!",
    "Ты снова здесь! Что будем делать?",
    "Люблю тебя! Как настроение?",
    "Ты как свежий воздух, я рад тебя видеть!",
    "Как же я тебя ждал, моя дорогая!",
    "Каждый раз, когда ты приходишь, это праздник!",
    "Ты - свет в моем окне!",
    "Как прекрасно, что ты вернулась!",
    "С тобой всегда радость и счастье!",
    "Какой сюрприз! Ты моя радость!",
    "Добрый день, красавица!",
    "Я так рад, что ты здесь!",
    "Как же я скучал по тебе!",
    "Ты делаешь каждый мой день лучше!"
]


# Функция для произнесения случайной фразы
def greet_wife():
    global LAST_GREET_TIME
    if time.time() - LAST_GREET_TIME > GREETING_INTERVAL:
        LAST_GREET_TIME = time.time()
        phrase = random.choice(greetings)  # Выбираем случайное приветствие
        threading.Thread(target=speak, args=(phrase,)).start()


# Функция для синтеза речи
def speak(text):
    engine.say(text)
    engine.runAndWait()


# Функция для анализа кадра
def process_frame(frame):
    try:
        # Конвертация кадра в чёрно-белый для ускорения обработки
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame is None or frame.size == 0:
            print("Кадр пустой!")
            return

        # Обнаружение лиц с помощью OpenCV
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if face_cascade.empty():
            raise Exception(
                "Не удалось загрузить классификатор! "
                "Проверьте путь к haarcascade_frontalface_default.xml"
            )

        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1,
                                              minNeighbors=3, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]  # Выделяем лицо

            for wife_image_path in wife_image_paths:
                result = DeepFace.verify(face, wife_image_path,
                                         enforce_detection=False)
                if result["verified"]:
                    greet_wife()
                    print("Жена обнаружена! Приветствую...")
                    return  # Прерываем, если нашли совпадение
    except Exception as e:
        print(f"Ошибка при распознавании: {e}")


# Открытие камеры
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()

    if ret:
        # Обрабатываем каждый 10-й кадр
        if int(time.time() * 10) % 10 == 0:
            threading.Thread(target=process_frame,
                             args=(frame.copy(),)).start()

        # Отображаем кадр
        cv2.imshow("Camera Feed", frame)

    # Выход при нажатии 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
