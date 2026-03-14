import torch
import torch.nn as nn

# Import 2 module anh em từ 2 thư mục
from src.vision.cnn_model import NewsImageExtractor
from src.nlp.nlp_model import NewsTextExtractor


class MultimodalNewsClassifier(nn.Module):
    """
    Sếp Tổng: Kêu gọi cả CNN và BiLSTM làm việc, sau đó tổng hợp kết quả.
    """

    def __init__(self, vocab_size, num_classes, dropout=0.3):
        super(MultimodalNewsClassifier, self).__init__()

        # 1. Khởi tạo 2 cỗ máy trích xuất
        # Gọi CNN tự build (output mặc định 512)
        self.image_extractor = NewsImageExtractor(output_dim=512, dropout=dropout)

        # Gọi NLP tự build (BiLSTM 256*2 = 512 chiều)
        self.text_extractor = NewsTextExtractor(
            vocab_size=vocab_size,
            hidden_dim=256,
            dropout=dropout,
        )

        # 2. Lớp Classifier tổng hợp
        # Kích thước = Vector Ảnh (512) + Vector Text (512) = 1024
        fusion_dim = self.image_extractor.output_dim + self.text_extractor.output_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, images, text_tensor, text_lengths):
        # 1. Hai nhân viên chạy song song
        image_vector = self.image_extractor(images)  # -> [Batch, 512]
        text_vector = self.text_extractor(text_tensor, text_lengths)  # -> [Batch, 512]

        # 2. Hợp nhất (Concatenate) dọc theo chiều features
        combined_vector = torch.cat(
            (image_vector, text_vector), dim=1
        )  # -> [Batch, 1024]

        # 3. Phân loại cuối cùng
        logits = self.classifier(combined_vector)

        return logits
