import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset


class TransformerNewsDataset(Dataset):
    def __init__(self, data_dir, transform, tokenizer, max_len=256):
        self.text_dir = os.path.join(data_dir, "texts")
        self.image_dir = os.path.join(data_dir, "images")
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.samples = []
        self.labels = []

        # Tự động nhận diện nhãn
        self.classes = sorted(
            list(
                set(
                    [
                        f.split("_")[0]
                        for f in os.listdir(self.text_dir)
                        if f.endswith(".json")
                    ]
                )
            )
        )
        self.category_to_id = {cat: idx for idx, cat in enumerate(self.classes)}
        self.id_to_category = {idx: cat for cat, idx in self.category_to_id.items()}

        for filename in os.listdir(self.text_dir):
            if filename.endswith(".json"):
                base_name = filename.replace(".json", "")
                img_path = os.path.join(self.image_dir, f"{base_name}.jpg")
                if os.path.exists(img_path):
                    with open(
                        os.path.join(self.text_dir, filename), "r", encoding="utf-8"
                    ) as f:
                        cat = json.load(f)["category"]
                    self.samples.append(
                        (os.path.join(self.text_dir, filename), img_path)
                    )
                    self.labels.append(self.category_to_id[cat])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text_path, img_path = self.samples[idx]

        # 1. Nhánh Vision: Xử lý ảnh
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)

        # 2. Nhánh NLP: Đọc text và dùng AutoTokenizer
        with open(text_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        full_text = f"{data.get('title', '')}. {data.get('content', '')}"

        # Hàm encode_plus thần thánh: Tự động thêm [CLS], [SEP], padding và tạo Attention Mask
        encoding = self.tokenizer.encode_plus(
            full_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        label_id = self.category_to_id[data["category"]]

        # Trả về một Dictionary cực kỳ chuyên nghiệp
        return {
            "image": image_tensor,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label_id, dtype=torch.long),
        }
