import os
import sys
import json
import random
import logging
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from transformers import AutoTokenizer  # <--- Vũ khí mới

# Import từ các file _transformer vừa tạo
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from src.data.dataset_transformer import TransformerNewsDataset
from src.pipeline.multimodal_transformer import PhoBertResNetFusion

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# --- MLOps: Early Stopping ---
class EarlyStopping:
    def __init__(
        self, patience=3, min_delta=0.005
    ):  # Patience = 3 vì Transformer hội tụ rất nhanh
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            logger.warning(f"⚠️ Early Stopping Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# KHỞI CHẠY (MAIN)
# ==========================================
def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Khởi động Fine-Tuning Transformer trên: {device.type.upper()}")

    # CẤU HÌNH (Rất quan trọng)
    RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "transformer_model.pth")
    BATCH_SIZE = 16  # VRAM GPU sẽ cạn nhanh với ResNet+PhoBERT, khuyên dùng 16
    MAX_EPOCHS = 10  # Chỉ cần 10, thực tế epoch 3-4 là đã max
    LEARNING_RATE = 2e-5  # BẮT BUỘC NHỎ (Fine-tuning không được dùng 1e-3)
    MAX_SEQ_LEN = 256  # 256 là đủ chuẩn cho tin tức báo chí

    # 1. Khởi tạo Tokenizer
    logger.info("Tải PhoBERT Tokenizer từ Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

    # 2. Dataset & Transforms
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    full_train_ds = TransformerNewsDataset(
        RAW_DATA_DIR, train_transform, tokenizer, MAX_SEQ_LEN
    )
    full_val_ds = TransformerNewsDataset(
        RAW_DATA_DIR, val_transform, tokenizer, MAX_SEQ_LEN
    )

    # 3. Stratified Split
    indices, labels = list(range(len(full_train_ds))), full_train_ds.labels
    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, labels, test_size=0.3, stratify=labels, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=42
    )

    train_loader = DataLoader(
        Subset(full_train_ds, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(Subset(full_val_ds, val_idx), batch_size=BATCH_SIZE)
    test_loader = DataLoader(Subset(full_val_ds, test_idx), batch_size=BATCH_SIZE)

    # 4. Khởi tạo Mô hình & Optimizer
    logger.info("Khởi tạo PhoBert + ResNet50...")
    model = PhoBertResNetFusion(num_classes=len(full_train_ds.classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=3)

    # 5. Huấn luyện (Fine-tuning Loop)
    logger.info("🔥 BẮT ĐẦU ÉP XUNG MÔ HÌNH (FINE-TUNING) 🔥")
    best_val_loss = float("inf")

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for batch in train_loader:
            imgs = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            masks = batch["attention_mask"].to(device)
            lbls = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(imgs, input_ids, masks)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (logits.argmax(1) == lbls).sum().item()
            train_total += lbls.size(0)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                masks = batch["attention_mask"].to(device)
                lbls = batch["label"].to(device)

                logits = model(imgs, input_ids, masks)
                val_loss += criterion(logits, lbls).item()
                val_correct += (logits.argmax(1) == lbls).sum().item()
                val_total += lbls.size(0)

        avg_val_loss = val_loss / len(val_loader)
        logger.info(
            f"Epoch {epoch+1:02d} | Train Acc: {train_correct/train_total*100:.2f}% "
            f"| Val Acc: {val_correct/val_total*100:.2f}% | Val Loss: {avg_val_loss:.4f}"
        )

        early_stopping(avg_val_loss)
        if early_stopping.counter == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "category_to_id": full_train_ds.category_to_id,
                    "id_to_category": full_train_ds.id_to_category,
                },
                MODEL_SAVE_PATH,
            )
            logger.info("   🌟 Đã lưu Model Fine-tune tốt nhất!")

        if early_stopping.early_stop:
            logger.info("🛑 Kích hoạt Early Stopping! Dừng Fine-tuning.")
            break

    # 6. Kiểm thử cuối cùng
    logger.info("🚀 BÀI THI CUỐI KỲ TRÊN TẬP TEST 🚀")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH)["model_state_dict"])
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            imgs, input_ids, masks, lbls = (
                batch["image"].to(device),
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["label"].to(device),
            )
            test_correct += (
                (model(imgs, input_ids, masks).argmax(1) == lbls).sum().item()
            )
            test_total += lbls.size(0)

    logger.info(f"🎯 KẾT QUẢ ĐỘ CHÍNH XÁC PHO-BERT: {test_correct/test_total*100:.2f}%")


if __name__ == "__main__":
    main()
