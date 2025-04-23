import os
import cv2
import time
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sqlalchemy import create_engine, Column, Integer, DateTime, String
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import desc, asc
from ultralytics import YOLO
import logging

# Настройка логирования
LOG_FILE = "smoking_detection.log"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
logging.info("Приложение запущено")

Base = declarative_base()

class CigaretteDetect(Base):
    """Модель для хранения информации об обнаружении сигарет в базе данных."""
    __tablename__ = 'cigarette_detect'

    id = Column(Integer, primary_key=True)
    timestamp_detect = Column(DateTime)
    image_detect = Column(String)
    smoking_label = Column(String)

    def __repr__(self):
        return f"<CigaretteDetect(timestamp_detect='{self.timestamp_detect}', image_detect='{self.image_detect}', smoking_label='{self.smoking_label}')>"

class PoseDetectionApp:
    """Приложение для обнаружения поз людей и сигарет на видео с камеры."""

    def __init__(self, window, window_title, yolopose_model_path, cigarette_model_path, classification_model_path,
                 output_folder="smoking_detect", db_url="postgresql://your_db_user:your_db_password@localhost/your_db_name"):
        """Инициализация приложения."""
        logging.info("Инициализация приложения PoseDetectionApp")
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1000x700")
        self.window.resizable(False, False)

        # Цветовая схема (темно-серый)
        self.bg_color = "#333333"
        self.fg_color = "#FFFFFF"
        self.button_bg = "#DDDDDD"
        self.button_fg = "#FFFFFF"
        self.button_hover_bg = "#444444"
        self.canvas_bg = "#222222"

        self.window.configure(bg=self.bg_color)

        self.yolopose_model_path = yolopose_model_path
        self.cigarette_model_path = cigarette_model_path
        self.classification_model_path = classification_model_path
        self.output_folder = output_folder
        self.db_url = db_url

        self.yolopose_model = self._load_model(yolopose_model_path, "YOLO Pose")
        self.cigarette_model = self._load_model(cigarette_model_path, "Cigarette YOLO")
        self.classification_model = self._load_classification_model(classification_model_path)

        self.cap = None
        self.is_running = False

        os.makedirs(self.output_folder, exist_ok=True)
        logging.info(f"Папка для вывода создана или уже существует: {self.output_folder}")

        self.engine = self._connect_to_db(db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        # UI Elements
        self.main_frame = tk.Frame(window, bg=self.bg_color)
        self.main_frame.pack(expand=True, fill=tk.BOTH, pady=(100, 0))

        self.canvas = tk.Canvas(self.main_frame, width=640, height=480, bg=self.canvas_bg, highlightthickness=0)
        self.canvas.pack(pady=10)

        self.button_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        self.button_frame.pack(pady=10)

        self.btn_start = self._create_button(self.button_frame, "Начать", self.start_detection)
        self.btn_stop = self._create_button(self.button_frame, "Остановить", self.stop_detection)
        self.btn_view_db = self._create_button(self.button_frame, "Просмотр БД", self.open_db_window)

        self.photo = None
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)  # Добавляем обработчик закрытия окна
        self.window.mainloop()
        logging.info("GUI приложения запущено")

    def _load_model(self, model_path, model_name):
        """Загружает YOLO модель."""
        try:
            model = YOLO(model_path)
            logging.info(f"Модель {model_name} успешно загружена из {model_path}")
            return model
        except Exception as e:
            logging.error(f"Ошибка при загрузке модели {model_name}: {e}")
            return None

    def _load_classification_model(self, model_path):
        """Загружает модель классификации."""
        try:
            model = tf.keras.models.load_model(model_path)
            logging.info(f"Модель классификации успешно загружена из {model_path}")
            return model
        except Exception as e:
            logging.error(f"Ошибка при загрузке модели классификации: {e}")
            return None

    def _connect_to_db(self, db_url):
        """Подключается к базе данных."""
        try:
            engine = create_engine(db_url)
            logging.info(f"Успешно подключились к базе данных по адресу: {db_url}")
            return engine
        except Exception as e:
            logging.error(f"Ошибка при подключении к базе данных: {e}")
            return None

    def _create_button(self, parent, text, command):
        """Создает кнопку с заданным стилем."""
        button = tk.Button(parent, text=text, width=15, command=command, bg=self.button_bg, fg=self.button_fg,
                           font=('Arial', 12, 'bold'), relief=tk.FLAT)
        button.pack(side=tk.LEFT, padx=10)
        button.bind("<Enter>", lambda event: button.config(bg=self.button_hover_bg))
        button.bind("<Leave>", lambda event: button.config(bg=self.button_bg, fg=self.button_fg))
        return button

    def start_detection(self):
        """Запускает процесс обнаружения с камеры."""
        logging.info("Запущена детекция")
        if not self.is_running:
            self.is_running = True
            self.cap = cv2.VideoCapture(0)

            if not self.cap.isOpened():
                logging.error("Ошибка: Не удалось открыть камеру.")
                self.is_running = False
                return

            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            threading.Thread(target=self.detection_loop, daemon=True).start()
            logging.info("Поток детекции запущен")

    def stop_detection(self):
        """Останавливает процесс обнаружения и освобождает ресурсы."""
        logging.info("Остановлена детекция")
        if self.is_running:
            self.is_running = False
            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)

            if self.cap and self.cap.isOpened():
                self.cap.release()
                logging.info("Камера освобождена")
            self.clear_canvas()

    def clear_canvas(self):
        """Очищает содержимое canvas."""
        self.canvas.delete("all")
        self.photo = None
        logging.info("Canvas очищен")

    def process_frame(self, frame):
        """Обрабатывает один кадр, обнаруживая позы людей и сигареты."""
        logging.debug("Обработка нового кадра")
        try:
            person_results = self.yolopose_model(frame, conf=0.5, save=False)
            logging.debug("Обнаружение поз людей выполнено")
            cigarette_results = self.cigarette_model(frame, conf=0.3, save=False)
            logging.debug("Обнаружение сигарет выполнено")

            return frame, person_results, cigarette_results
        except Exception as e:
            logging.error(f"Ошибка при обработке кадра: {e}")
            return frame, None, None

    def is_cigarette_in_person_box(self, person_box, cigarette_box):
        """Проверяет, находится ли сигарета в зоне человека."""
        px1, py1, px2, py2 = person_box
        cx1, cy1, cx2, cy2 = cigarette_box

        # Проверяем, пересекаются ли прямоугольники
        result = px1 < cx2 and px2 > cx1 and py1 < cy2 and py2 > cy1
        logging.debug(f"Проверка нахождения сигареты в зоне человека: {result}")
        return result

    def classify_image(self, image_path):
        """Классифицирует изображение как "smoking" или "nonsmoking" с помощью модели."""
        try:
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.

            prediction = self.classification_model.predict(img_array)
            result = "smoking" if prediction[0][0] > 0.5 else "nonsmoking"
            logging.info(f"Изображение классифицировано как: {result}")
            return result
        except Exception as e:
            logging.error(f"Ошибка при классификации изображения: {e}")
            return "nonsmoking"  # Возвращаем значение по умолчанию

    def save_smoking_image(self, frame):
        """Сохраняет исходное изображение с меткой о курении."""
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_folder, f"smoking_{timestamp}.jpg")
        try:
            cv2.imwrite(filename, frame)
            logging.info(f"Изображение сохранено: {filename}")
            smoking_label = self.classify_image(filename)

            self.insert_data_to_db(timestamp, filename, smoking_label)
            logging.info(f"Сохранено изображение: {filename}, Классификация: {smoking_label}")

        except Exception as e:
            logging.error(f"Ошибка при сохранении изображения: {e}. Путь сохранения: {filename}")

    def insert_data_to_db(self, timestamp, image_path, smoking_label):
        """Вставляет данные об обнаружении в базу данных."""
        try:
            cigarette_detect = CigaretteDetect(timestamp_detect=datetime.strptime(timestamp, "%Y%m%d_%H%M%S"),
                                                image_detect=image_path,
                                                smoking_label=smoking_label)
            self.session.add(cigarette_detect)
            self.session.commit()
            logging.info("Данные успешно добавлены в базу данных")

        except Exception as e:
            logging.error(f"Ошибка при добавлении данных в базу данных: {e}")
            self.session.rollback()

    def paint(self, image, person_results, cigarette_results, person_color=(255, 0, 255), cigarette_color=(0, 255, 0), line_thickness=2):
        """Рисует скелеты людей и рамки сигарет на изображении."""
        skeleton_image = image.copy()

        person_boxes = []
        if person_results and len(person_results) > 0:
            skeleton_image = person_results[0].plot()
            for box in person_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                person_boxes.append((x1, y1, x2, y2))

        if cigarette_results and len(cigarette_results) > 0:
            for result in cigarette_results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0].item()
                    label_text = f"cigarette: {conf:.2f}"

                    cv2.rectangle(skeleton_image, (x1, y1), (x2, y2), cigarette_color, line_thickness)
                    cv2.putText(skeleton_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cigarette_color, line_thickness)

                    if conf > 0.70:
                        cigarette_box = (x1, y1, x2, y2)
                        for person_box in person_boxes:
                            if self.is_cigarette_in_person_box(person_box, cigarette_box):
                                logging.info("Сигарета обнаружена в зоне человека. Изображение будет сохранено.")
                                return skeleton_image, True

        return skeleton_image, False

    def detection_loop(self):
        """Основной цикл обнаружения, который считывает кадры с камеры, обрабатывает их и отображает результат."""
        logging.info("Запущен цикл детекции")
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                logging.error("Ошибка: Не удалось получить кадр с камеры.")
                self.stop_detection()
                break

            processed_frame, person_results, cigarette_results = self.process_frame(frame)

            # Добавлена проверка на None для результатов
            if person_results is not None and cigarette_results is not None:
                painted_frame, should_save = self.paint(processed_frame, person_results, cigarette_results)
            else:
                painted_frame = processed_frame  # Если результатов нет, используем исходный кадр
                should_save = False

            if should_save:
                self.save_smoking_image(frame)

            frame_rgb = cv2.cvtColor(painted_frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            self.window.update()
            time.sleep(0.01)

    def open_db_window(self):
        """Открывает новое окно для просмотра изображений из базы данных."""
        logging.info("Открыто окно просмотра базы данных")
        db_window = tk.Toplevel(self.window, bg=self.bg_color)
        db_window.title("Просмотр изображений из базы данных")
        db_window.geometry("1200x800")
        db_window.resizable(False, False)
        DatabaseViewer(db_window, self.session, self.bg_color, self.fg_color, self.button_bg, self.button_hover_bg)

    def on_closing(self):
        """Обработчик закрытия окна."""
        logging.info("Закрытие приложения")
        try:
            if hasattr(self, 'session') and self.session is not None:
                self.session.close()
                logging.info("Сессия SQLAlchemy закрыта")
        except Exception as e:
            logging.error(f"Ошибка при закрытии сессии SQLAlchemy: {e}")
        finally:
            self.window.destroy()
            logging.info("Окно приложения закрыто")

class DatabaseViewer:
    """Класс для просмотра изображений из базы данных."""

    def __init__(self, window, session, bg_color, fg_color, button_bg, button_hover_bg):
        """Инициализация просмотрщика базы данных."""
        logging.info("Инициализация DatabaseViewer")
        self.window = window
        self.window.title("Детекты")
        self.window.configure(bg=bg_color)

        self.session = session
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.button_bg = button_bg
        self.button_hover_bg = button_hover_bg

        self.sort_order = tk.StringVar(value="Новые")
        self.filter_label = tk.StringVar(value="Все")
        self.current_image_index = 0
        self.image_paths = self.load_image_paths()

        # Основной фрейм для организации зон
        main_frame = tk.Frame(window, bg=bg_color)
        main_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Левая зона (список файлов)
        left_frame = tk.Frame(main_frame, bg=bg_color)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=(40, 0))

        self.listbox = tk.Listbox(left_frame, width=30, height=20, bg=self.bg_color, fg=self.fg_color, borderwidth=0,
                                  highlightthickness=0, font=('Arial', 10))
        self.listbox.pack(pady=10, padx=10, side=tk.LEFT, fill=tk.Y)
        for path in self.image_paths:
            self.listbox.insert(tk.END, os.path.basename(path))

        # Центральная зона (отображение изображения)
        center_frame = tk.Frame(main_frame, bg=bg_color)
        center_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(center_frame, bg=self.bg_color, fg=self.fg_color)
        self.image_label.pack(pady=(10, 0), padx=10, fill=tk.BOTH, expand=True)
        self.image_label.image = None

        # Нижняя зона (кнопки и фильтры)
        bottom_frame = tk.Frame(window, bg=bg_color)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        # Фрейм для кнопок навигации
        navigation_frame = tk.Frame(bottom_frame, bg=bg_color)
        navigation_frame.pack(side=tk.LEFT, padx=10)

        self.prev_button = tk.Button(navigation_frame, text="Предыдущее", command=self.show_previous_image, bg=self.button_bg,
                                     fg=self.fg_color, font=('Arial', 12, 'bold'), relief=tk.FLAT)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        self.prev_button.bind("<Enter>", lambda event: self.prev_button.config(bg=self.button_hover_bg, fg=self.fg_color))
        self.prev_button.bind("<Leave>", lambda event: self.prev_button.config(bg=self.button_bg, fg=self.fg_color))

        self.next_button = tk.Button(navigation_frame, text="Следующее", command=self.show_next_image, bg=self.button_bg,
                                     fg=self.fg_color, font=('Arial', 12, 'bold'), relief=tk.FLAT)
        self.next_button.pack(side=tk.LEFT, padx=5)
        self.next_button.bind("<Enter>", lambda event: self.next_button.config(bg=self.button_hover_bg, fg=self.fg_color))
        self.next_button.bind("<Leave>", lambda event: self.next_button.config(bg=self.button_bg, fg=self.fg_color))

        # Фрейм для фильтров и сортировки
        filter_frame = tk.Frame(bottom_frame, bg=bg_color)
        filter_frame.pack(side=tk.RIGHT, padx=10)

        # Sort ComboBox
        self.sort_options = ["Новые", "Старые"]
        self.sort_combo = ttk.Combobox(filter_frame, textvariable=self.sort_order, values=self.sort_options,
                                       state="readonly", font=('Arial', 10), background=self.bg_color,
                                       foreground=self.fg_color)
        self.sort_combo.pack(side=tk.LEFT, padx=5)
        self.sort_combo.bind("<<ComboboxSelected>>", self.sort_images)

        # Filter ComboBox
        self.filter_options = ["Все", "smoking", "nonsmoking"]
        self.filter_combo = ttk.Combobox(filter_frame, textvariable=self.filter_label, values=self.filter_options,
                                         state="readonly", font=('Arial', 10), background=self.bg_color,
                                         foreground=self.fg_color)
        self.filter_combo.pack(side=tk.LEFT, padx=5)
        self.filter_combo.bind("<<ComboboxSelected>>", self.filter_images)

        # Status Label
        self.status_label = tk.Label(center_frame, text="", bg=self.bg_color, fg=self.fg_color, font=('Arial', 9))
        self.status_label.pack(pady=5, fill=tk.X)

        self.listbox.bind("<Double-Button-1>", self.show_selected_image)

        if self.image_paths:
            self.show_image(self.image_paths[0])

        # Стиль для комбобоксов
        style = ttk.Style()
        style.configure("TCombobox", background=self.bg_color, foreground=self.fg_color)

    def load_image_paths(self):
        """Загружает пути к изображениям из базы данных, сортируя и фильтруя их."""
        selected_sort_order = self.sort_order.get()
        selected_filter = self.filter_label.get()
        logging.info(f"Загрузка путей к изображениям: Сортировка = {selected_sort_order}, Фильтр = {selected_filter}")
        try:
            query = self.session.query(CigaretteDetect)

            if selected_filter != "Все":
                query = query.filter(CigaretteDetect.smoking_label == selected_filter)
                logging.debug(f"Применен фильтр: {selected_filter}")

            if selected_sort_order == "Новые":
                images = query.order_by(desc(CigaretteDetect.timestamp_detect)).all()
                logging.debug("Сортировка по новым")
            else:
                images = query.order_by(asc(CigaretteDetect.timestamp_detect)).all()
                logging.debug("Сортировка по старым")

            image_paths = [image.image_detect for image in images]
            logging.info(f"Загружено {len(image_paths)} путей к изображениям из базы данных.")
            return image_paths
        except Exception as e:
            logging.error(f"Ошибка при загрузке путей к изображениям: {e}")
            return []

    def show_image(self, image_path):
        """Отображает изображение по указанному пути."""
        logging.info(f"Отображение изображения: {image_path}")
        try:
            image = Image.open(image_path)
            image = image.resize((700, 500), Image.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.tk_image)
            self.image_label.image = self.tk_image
            self.update_status_label(image_path)  # Передаем путь изображения в update_status_label
        except Exception as e:
            logging.error(f"Не удалось открыть изображение: {image_path}\nОшибка: {e}")
            self.image_label.config(text=f"Не удалось открыть изображение: {image_path}\nОшибка: {e}")

    def show_selected_image(self, event):
        """Отображает выбранное изображение из списка."""
        selected_index = self.listbox.curselection()
        if selected_index:
            self.current_image_index = selected_index[0]
            image_path = self.image_paths[selected_index[0]]
            self.show_image(image_path)
            logging.info(f"Пользователь выбрал изображение: {image_path}")

    def show_next_image(self):
        """Отображает следующее изображение в списке."""
        if self.image_paths:
            self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
            self.show_image(self.image_paths[self.current_image_index])
            logging.info("Показано следующее изображение")

    def show_previous_image(self):
        """Отображает предыдущее изображение в списке."""
        if self.image_paths:
            self.current_image_index = (self.current_image_index - 1) % len(self.image_paths)
            self.show_image(self.image_paths[self.current_image_index])
            logging.info("Показано предыдущее изображение")

    def sort_images(self, event=None):
        """Сортирует изображения в списке."""
        self.image_paths = self.load_image_paths()
        self.current_image_index = 0

        self.listbox.delete(0, tk.END)
        for path in self.image_paths:
            self.listbox.insert(tk.END, os.path.basename(path))

        if self.image_paths:
            self.show_image(self.image_paths[0])
        self.update_status_label(self.image_paths[0] if self.image_paths else None)

        logging.info("Изображения отсортированы")

    def filter_images(self, event=None):
        """Фильтрует изображения в списке."""
        self.image_paths = self.load_image_paths()
        self.current_image_index = 0

        self.listbox.delete(0, tk.END)
        for path in self.image_paths:
            self.listbox.insert(tk.END, os.path.basename(path))

        if self.image_paths:
            self.show_image(self.image_paths[0])
        self.update_status_label(self.image_paths[0] if self.image_paths else None)

        logging.info("Изображения отфильтрованы")

    def update_status_label(self, image_path=None):
        """Обновляет метку статуса с информацией о текущем изображении."""
        if image_path:
            image_name = os.path.basename(image_path)

            try:
                cigarette_detect = self.session.query(CigaretteDetect).filter_by(image_detect=image_path).first()
                if cigarette_detect:
                    smoking_label = cigarette_detect.smoking_label
                    status_text = f"Название изображения: {image_name} ({smoking_label})"
                else:
                    status_text = f"Название изображения: {image_name} (Метка отсутствует)"
                logging.debug(f"Статус метки обновлен: {status_text}")
            except Exception as e:
                status_text = f"Название изображения: {image_name} (Ошибка при загрузке метки)"
                logging.error(f"Ошибка при получении метки из базы данных: {e}")

            self.status_label.config(text=status_text, bg=self.bg_color, fg=self.fg_color)
        else:
            self.status_label.config(text="Нет изображений", bg=self.bg_color, fg=self.fg_color)
            logging.info("Нет изображений для отображения")

if __name__ == "__main__":
    root = tk.Tk()
    app = PoseDetectionApp(root, "Smoking Detect",
                                    yolopose_model_path="C:/Users/Максим/PycharmProjects/PythonProject/yolov8s-pose.pt",
                                    cigarette_model_path="C:/Users/Максим/PycharmProjects/PythonProject/v8s.pt",
                                    classification_model_path="C:/Users/Максим/PycharmProjects/PythonProject/smoking_detection_model.h5",
                                    output_folder="smoking_detect",
                                    db_url="postgresql://postgres:12345@localhost/smoking_detection")

    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    logging.info("Приложение завершено")