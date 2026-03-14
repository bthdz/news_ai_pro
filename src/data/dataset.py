import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image


class NewsDataset(Dataset):
    """Dataset kết hợp Ảnh và Text, có lọc dữ liệu lỗi."""

    def __init__(self, data_dir, transform=None, text_processor=None):
        self.text_dir = os.path.join(data_dir, "texts")
        self.image_dir = os.path.join(data_dir, "images")
        self.transform = transform
        self.text_processor = text_processor

        if not os.path.exists(self.text_dir) or not os.path.exists(self.image_dir):
            raise FileNotFoundError(
                "Không tìm thấy thư mục data! Hãy chạy scraper.py trước."
            )

        print("🔍 Đang quét và kiểm tra dữ liệu...")
        all_json_files = [f for f in os.listdir(self.text_dir) if f.endswith(".json")]
        self.samples = []
        unique_categories = set()

        for filename in all_json_files:
            json_path = os.path.join(self.text_dir, filename)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                category = data.get("category")
                image_file = data.get("image_file")
                image_path = os.path.join(self.image_dir, image_file)

                if category and os.path.exists(image_path):
                    unique_categories.add(category)
                    self.samples.append(filename)
            except Exception:
                pass

        self.classes = sorted(list(unique_categories))
        self.category_to_id = {cat: idx for idx, cat in enumerate(self.classes)}
        self.id_to_category = {idx: cat for idx, cat in enumerate(self.classes)}
        print(
            f"Đã nhận diện {len(self.samples)} bài báo hợp lệ và {len(self.classes)} nhãn."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        json_file = self.samples[idx]
        json_path = os.path.join(self.text_dir, json_file)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        label_id = self.category_to_id[data.get("category")]

        # --- 1. XỬ LÝ TEXT ---
        title = data.get("title", "").strip()
        content = data.get("content", "").strip()
        text_content = f"{title}. {content}"

        if self.text_processor:
            text_tensor, text_length = self.text_processor.text_to_tensor(text_content)
        else:
            text_tensor = torch.zeros(200, dtype=torch.long)
            text_length = torch.tensor(1, dtype=torch.long)

        # --- 2. XỬ LÝ ẢNH ---
        image_name = data.get("image_file")
        image_path = os.path.join(self.image_dir, image_name)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color="black")

        if self.transform:
            image = self.transform(image)
        else:
            from torchvision import transforms

            image = transforms.ToTensor()(image)

        return image, text_tensor, text_length, torch.tensor(label_id, dtype=torch.long)
