import os
import sys
import json
import random
import logging
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# ==========================================
# 0. SETUP HỆ THỐNG & ĐƯỜNG DẪN
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

# Import các module (ĐẢM BẢO BẠN ĐÃ CẬP NHẬT CÁC FILE NÀY GIỐNG KAGGLE)
from src.utils.preprocess import TextProcessor
from src.data.dataset import NewsDataset
from src.pipeline.multimodal import MultimodalNewsClassifier

# Cấu hình Logging chuyên nghiệp
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """Cố định seed để đảm bảo tái lập kết quả 100%"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================================
# UTILS: MLOPS TOOLS (EARLY STOPPING & PHÂN TÍCH DATA)
# ==========================================
class EarlyStopping:
    """Công cụ 'canh gác' tránh Overfitting"""

    def __init__(self, patience=5, min_delta=0.001):
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


def analyze_corpus(text_dir):
    """Phân tích văn bản để tìm ra MAX_SEQ_LEN tối ưu dựa trên phân vị 95"""
    logger.info("Đang phân tích thống kê độ dài các bài báo...")
    lengths, all_texts = [], []

    if not os.path.exists(text_dir):
        logger.error(f"Không tìm thấy thư mục: {text_dir}")
        sys.exit(1)

    for f_name in os.listdir(text_dir):
        if f_name.endswith(".json"):
            with open(os.path.join(text_dir, f_name), "r", encoding="utf-8") as f:
                d = json.load(f)
                full_text = f"{d.get('title', '')}. {d.get('content', '')}"
                words = full_text.lower().split()
                lengths.append(len(words))
                all_texts.append(full_text)

    p95 = np.percentile(lengths, 95)
    optimal_max_len = int(p95)

    logger.info(
        f"📊 Trung bình: {np.mean(lengths):.0f} từ/bài | 95% bài báo < {optimal_max_len} từ."
    )
    return optimal_max_len, all_texts


# ==========================================
# 1. CẤU HÌNH DỰ ÁN (CONFIG)
# ==========================================
class Config:
    RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "multimodal_model.pth")

    BATCH_SIZE = 32  # Đã tăng lên 32 cho data lớn
    MAX_EPOCHS = 50  # Tăng lên 50, Early Stopping sẽ tự lo điểm dừng
    LEARNING_RATE = 1e-3
    MAX_VOCAB_SIZE = 30000
    SEED = 42

    # Tham số Early Stopping & Scheduler
    PATIENCE = 5
    MIN_DELTA = 0.005


def evaluate(model, dataloader, criterion, device):
    """Hàm đánh giá dùng chung cho Validation và Test"""
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, texts_tensor, text_lengths, labels in dataloader:
            images = images.to(device)
            texts_tensor = texts_tensor.to(device)
            text_lengths = text_lengths.to(device)
            labels = labels.to(device)

            logits = model(images, texts_tensor, text_lengths)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = (correct / total) * 100 if total > 0 else 0
    return avg_loss, accuracy


# ==========================================
# 2. HÀM CHẠY CHÍNH (MAIN WORKFLOW)
# ==========================================
def main():
    config = Config()
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    set_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Khởi động luồng Huấn luyện trên thiết bị: {device.type.upper()}")

    # --- BƯỚC 1: XÂY DỰNG TỪ ĐIỂN ĐỘNG ---
    text_dir = os.path.join(config.RAW_DATA_DIR, "texts")
    optimal_max_len, all_texts = analyze_corpus(text_dir)

    processor = TextProcessor(
        max_vocab_size=config.MAX_VOCAB_SIZE, max_seq_len=optimal_max_len
    )
    processor.build_vocab(all_texts)
    logger.info(f"📚 Từ điển thực tế: {len(processor.word2idx)} từ.")

    # --- BƯỚC 2: KHỞI TẠO 2 BỘ TRANSFORM (DUAL INSTANTIATION) ---
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),  # Tăng cường data
            transforms.RandomRotation(15),  # Tăng cường data
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

    full_train_dataset = NewsDataset(config.RAW_DATA_DIR, train_transform, processor)
    full_val_dataset = NewsDataset(config.RAW_DATA_DIR, val_transform, processor)

    if len(full_train_dataset) == 0:
        logger.error("Dataset rỗng! Kiểm tra lại thư mục raw data.")
        sys.exit(1)

    # --- BƯỚC 3: STRATIFIED SPLIT (CHIA ĐỀU DATA 12 NHÃN) ---
    indices = list(range(len(full_train_dataset)))

    # Lấy labels để chia. Nếu dataset không có thuộc tính labels, ta tạo list tạm
    if hasattr(full_train_dataset, "labels"):
        labels = full_train_dataset.labels
    else:
        logger.warning("Dataset không có thuộc tính labels. Tự động nội suy labels...")
        labels = [
            full_train_dataset.category_to_id[json.load(open(sample[0]))["category"]]
            for sample in full_train_dataset.samples
        ]

    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, labels, test_size=0.3, stratify=labels, random_state=config.SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=config.SEED
    )

    logger.info(
        f"Đã chia Data: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}"
    )

    train_ds = Subset(full_train_dataset, train_idx)
    val_ds = Subset(full_val_dataset, val_idx)  # Val và Test phải dùng nguyên bản
    test_ds = Subset(full_val_dataset, test_idx)

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE)

    # --- BƯỚC 4: KHỞI TẠO MÔ HÌNH & OPTIMIZER CHUẨN MLOPS ---
    logger.info("Khởi tạo Mạng Đa Phương Thức (CNN + BiLSTM)...")
    model = MultimodalNewsClassifier(
        vocab_size=len(processor.word2idx), num_classes=len(full_train_dataset.classes)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4
    )  # Dùng AdamW chống Overfit

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    early_stopping = EarlyStopping(patience=config.PATIENCE, min_delta=config.MIN_DELTA)

    # --- BƯỚC 5: VÒNG LẶP HUẤN LUYỆN (TRAINING LOOP) ---
    logger.info(f"🔥 BẮT ĐẦU HUẤN LUYỆN (Tối đa {config.MAX_EPOCHS} Epochs) 🔥")

    for epoch in range(config.MAX_EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for images, texts_tensor, text_lengths, batch_labels in train_loader:
            images = images.to(device)
            texts_tensor = texts_tensor.to(device)
            text_lengths = text_lengths.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(images, texts_tensor, text_lengths)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(logits, 1)
            train_correct += (preds == batch_labels).sum().item()
            train_total += batch_labels.size(0)

        train_acc = (train_correct / train_total) * 100
        avg_train_loss = train_loss / len(train_loader)

        # Đánh giá Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        logger.info(
            f"Epoch [{epoch+1:02d}/{config.MAX_EPOCHS}] "
            f"| Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f}"
        )

        # Kích hoạt Scheduler & Early Stopping
        scheduler.step(val_loss)
        early_stopping(val_loss)

        if early_stopping.counter == 0:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "word2idx": processor.word2idx,
                "idx2word": processor.idx2word,
                "category_to_id": full_train_dataset.category_to_id,
                "id_to_category": full_train_dataset.id_to_category,
                "max_seq_len": optimal_max_len,
            }
            torch.save(checkpoint, config.MODEL_SAVE_PATH)
            logger.info("   🌟 Đã lưu Checkpoint (Model tốt nhất hiện tại)!")

        if early_stopping.early_stop:
            logger.info(
                "🛑 KÍCH HOẠT EARLY STOPPING! Đã tìm thấy điểm tối ưu. Dừng huấn luyện."
            )
            break

    # --- BƯỚC 6: BÀI THI CUỐI KỲ (TESTING) ---
    logger.info("🚀 BẮT ĐẦU KIỂM THỬ TRÊN TẬP TEST CHƯA TỪNG THẤY 🚀")
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH)["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    logger.info(f"=== KẾT QUẢ CUỐI CÙNG ===")
    logger.info(f"🎯 Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
    logger.info("=== HOÀN TẤT DỰ ÁN ===")


if __name__ == "__main__":
    main()
