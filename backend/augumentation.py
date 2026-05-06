import os

import cv2
import numpy as np
import random


class ImageAugmentor:
    """Аугментация изображений с пересчётом bounding boxes"""

    def augment(self, image_path: str, bboxes: list[tuple]) -> tuple:
        """
        Применяет 1-3 случайные трансформации к изображению.

        Args:
            image_path: путь к изображению
            bboxes: список боксов в YOLO-формате
        Returns:
            (augmented_image, augmented_bboxes)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"не удалось прочитать изображение: {image_path}")

        transforms = [
            self._adjust_brightness,
            self._adjust_contrast,
            self._gaussian_blur,
            self._gaussian_noise,
            self._horizontal_flip,
            self._channel_shuffle,
            self._color_jitter,
            self._sharpen,
            self._clahe,
            self._random_shadow,
            self._rotate_small,
        ]
        chosen = random.sample(transforms, k=random.randint(1, 3))

        new_bboxes = [bbox for bbox in bboxes]
        for transform in chosen:
            image, new_bboxes = transform(image, new_bboxes)

        return image, new_bboxes

    # ─── простые: боксы не меняются ───

    def _adjust_brightness(self, image, bboxes):
        """случайное изменение яркости"""
        beta = random.randint(-40, 40)
        image = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
        return image, bboxes

    def _adjust_contrast(self, image, bboxes):
        """случайное изменение контраста"""
        alpha = round(random.uniform(0.6, 1.4), 2)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        return image, bboxes

    def _gaussian_blur(self, image, bboxes):
        """гауссово размытие"""
        ksize = random.choice([3, 5, 7])
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        return image, bboxes

    def _gaussian_noise(self, image, bboxes):
        """гауссов шум"""
        sigma = random.randint(10, 30)
        noise = np.random.normal(0, sigma, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return image, bboxes

    def _channel_shuffle(self, image, bboxes):
        """перемешивание цветовых каналов (BGR -> случайный порядок)"""
        channels = list(range(3))
        random.shuffle(channels)
        image = image[:, :, channels]
        return image, bboxes

    def _color_jitter(self, image, bboxes):
        """сдвиг оттенка и насыщенности в HSV пространстве"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-15, 15)) % 180  # hue
        hsv[:, :, 1] = np.clip(
            hsv[:, :, 1] + random.randint(-30, 30), 0, 255
        )  # saturation
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return image, bboxes

    def _sharpen(self, image, bboxes):
        """повышение резкости"""
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image = cv2.filter2D(image, -1, kernel)
        return image, bboxes

    def _clahe(self, image, bboxes):
        """адаптивное выравнивание гистограммы (улучшение локального контраста)"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=random.uniform(1.5, 3.0), tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return image, bboxes

    def _random_shadow(self, image, bboxes):
        """случайная тень — затемнение случайной прямоугольной области"""
        h, w = image.shape[:2]
        x1 = random.randint(0, w // 2)
        y1 = random.randint(0, h // 2)
        x2 = random.randint(w // 2, w)
        y2 = random.randint(h // 2, h)
        darkness = random.uniform(0.3, 0.7)
        image = image.copy()
        image[y1:y2, x1:x2] = (image[y1:y2, x1:x2] * darkness).astype(np.uint8)
        return image, bboxes

    # ─── сложные: боксы пересчитываются ───

    def _horizontal_flip(self, image, bboxes):
        """горизонтальный flip — пересчёт x_center"""
        image = cv2.flip(image, 1)
        new_bboxes = []
        for bbox in bboxes:
            class_id, x_center, y_center, width, height = bbox
            new_bboxes.append((class_id, 1.0 - x_center, y_center, width, height))
        return image, new_bboxes

    def _rotate_small(self, image, bboxes):
        """поворот на небольшой угол (±15) — пересчёт боксов через матрицу поворота"""
        h, w = image.shape[:2]
        angle = random.uniform(-15, 15)
        center = (w / 2, h / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, matrix, (w, h), borderValue=(114, 114, 114))

        new_bboxes = []
        for bbox in bboxes:
            class_id, xc, yc, bw, bh = bbox

            # денормализуем
            abs_xc, abs_yc = xc * w, yc * h
            abs_w, abs_h = bw * w, bh * h

            # 4 угла бокса
            corners = np.array(
                [
                    [abs_xc - abs_w / 2, abs_yc - abs_h / 2],
                    [abs_xc + abs_w / 2, abs_yc - abs_h / 2],
                    [abs_xc + abs_w / 2, abs_yc + abs_h / 2],
                    [abs_xc - abs_w / 2, abs_yc + abs_h / 2],
                ]
            )

            # применяем матрицу поворота к каждому углу
            ones = np.ones((4, 1))
            corners_h = np.hstack([corners, ones])
            rotated = matrix.dot(corners_h.T).T

            # новый bounding box из повёрнутых углов
            new_x1 = max(0, rotated[:, 0].min())
            new_y1 = max(0, rotated[:, 1].min())
            new_x2 = min(w, rotated[:, 0].max())
            new_y2 = min(h, rotated[:, 1].max())

            # нормализуем обратно
            new_xc = ((new_x1 + new_x2) / 2) / w
            new_yc = ((new_y1 + new_y2) / 2) / h
            new_bw = (new_x2 - new_x1) / w
            new_bh = (new_y2 - new_y1) / h

            if new_bw > 0.01 and new_bh > 0.01:
                new_bboxes.append((class_id, new_xc, new_yc, new_bw, new_bh))

        return image, new_bboxes

    def augment_dataset(
        self,
        images_dir: str,
        labels_dir: str,
        output_images_dir: str,
        output_labels_dir: str,
        max_per_image: int = 2,
        sample_ratio: float = 0.75,
    ):
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)

        images = [
            f
            for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        selected = random.sample(images, k=int(len(images) * sample_ratio))

        generated = 0

        for img_name in selected:
            stem = os.path.splitext(img_name)[0]
            label_path = os.path.join(labels_dir, stem + ".txt")

            if not os.path.exists(label_path):
                continue

            # парсим боксы
            bboxes = []
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    bboxes.append(
                        (
                            int(parts[0]),
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3]),
                            float(parts[4]),
                        )
                    )

            image_path = os.path.join(images_dir, img_name)

            for i in range(max_per_image):
                aug_image, aug_bboxes = self.augment(image_path, bboxes)

                out_name = f"{stem}_aug{generated}"
                cv2.imwrite(
                    os.path.join(output_images_dir, out_name + ".jpg"), aug_image
                )

                with open(os.path.join(output_labels_dir, out_name + ".txt"), "w") as f:
                    for bbox in aug_bboxes:
                        f.write(
                            f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n"
                        )

                generated += 1

        return generated
