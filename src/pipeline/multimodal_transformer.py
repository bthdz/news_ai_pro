import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModel


class PhoBertResNetFusion(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()

        # --- NHÁNH 1: RESNET 50 (VISION) ---
        # Tải trọng số chuẩn từ ImageNet (Transfer Learning)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Cắt bỏ lớp Linear(fc) cuối cùng để lấy vector đặc trưng (2048 chiều)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])

        # --- NHÁNH 2: PHO-BERT (NLP) ---
        # Tải "bộ não" PhoBERT đã đọc 20GB tiếng Việt
        self.text_encoder = AutoModel.from_pretrained("vinai/phobert-base-v2")

        # --- HỢP NHẤT (FUSION) ---
        # 2048 (từ ResNet) + 768 (từ PhoBERT [CLS] token) = 2816
        fusion_dim = 2048 + 768

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, images, input_ids, attention_mask):
        # 1. Luồng Ảnh -> Vector (Batch, 2048)
        img_features = self.image_encoder(images)
        img_features = torch.flatten(img_features, 1)

        # 2. Luồng Chữ -> Vector (Batch, 768)
        # Lấy pooler_output (đại diện cho toàn bộ câu) từ mô hình Transformer
        text_outputs = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        text_features = text_outputs.pooler_output

        # 3. Nối 2 vector lại và Phân loại
        combined = torch.cat((img_features, text_features), dim=1)
        return self.classifier(combined)
