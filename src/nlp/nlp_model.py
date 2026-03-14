import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAttention(nn.Module):
    """
    Cơ chế Attention (Sự chú ý) tự viết từ đầu.
    Giúp mô hình biết nên tập trung vào từ khóa nào trong câu.
    """

    def __init__(self, hidden_dim):
        super(GlobalAttention, self).__init__()
        # Lớp tuyến tính để tính điểm số (Score) cho từng từ
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_outputs, mask=None):
        # lstm_outputs: [batch_size, seq_len, hidden_dim]

        # 1. Tính điểm số thô và loại bỏ chiều cuối để có shape [batch_size, seq_len]
        attn_weights = self.attention(lstm_outputs).squeeze(2)  # [batch_size, seq_len]

        # 2. Masking - Ép điểm của các số 0 (PAD - từ độn thêm) về âm vô cùng
        # Để khi qua Softmax, xác suất chú ý vào các số 0 này sẽ bằng đúng 0%
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == False, -1e9)

        # 3. Tính phân phối xác suất (tổng bằng 1)
        attn_weights = F.softmax(attn_weights, dim=1)

        # 4. Nhân trọng số với LSTM Output để tạo Vector Ngữ Cảnh (Context Vector)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_outputs).squeeze(1)

        # Trả về Context Vector và cả Trọng số để sau này có thể vẽ biểu đồ
        return context, attn_weights


class NewsTextExtractor(nn.Module):  # Đổi tên thành Extractor cho chuẩn nghĩa
    """
    Module NLP chuyên trách: Nhận Index Text -> Embedding -> BiLSTM -> Attention -> Context Vector
    """

    def __init__(
        self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3
    ):
        super(NewsTextExtractor, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = GlobalAttention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)

        self.output_dim = hidden_dim * 2  # Đầu ra luôn là 512 (256 * 2)

    def forward(self, text_tensor, text_lengths):
        mask = text_tensor != 0
        embedded = self.dropout(self.embedding(text_tensor))

        lengths_cpu = text_lengths.cpu() if text_lengths.is_cuda else text_lengths
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths_cpu, batch_first=True, enforce_sorted=False
        )

        packed_output, _ = self.lstm(packed_embedded)

        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=text_tensor.size(1)
        )

        # CHỈ TRẢ VỀ VECTOR NGỮ CẢNH [Batch, 512]
        context_vector, attn_weights = self.attention(lstm_out, mask)

        return context_vector


# # --- TEST NHANH KHI CHẠY TRỰC TIẾP FILE NÀY ---
# if __name__ == "__main__":
#     VOCAB_SIZE = 5000
#     NUM_CLASSES = 6
#     BATCH_SIZE = 2

#     model = NewsTextClassifier(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES)
#     # Giả lập 1 batch gồm 2 câu văn (câu 1 dài 4 từ, câu 2 dài 6 từ)
#     dummy_text = torch.tensor(
#         [[15, 42, 102, 9, 0, 0, 0, 0], [8, 21, 55, 11, 88, 12, 0, 0]]
#     )
#     dummy_lengths = torch.tensor([4, 6])

#     logits, attn = model(dummy_text, dummy_lengths)

#     print(
#         f"Shape Đầu ra (Dự đoán): {logits.shape} -> Chuẩn là [{BATCH_SIZE}, {NUM_CLASSES}]"
#     )
#     print(f"Shape Attention (Sự chú ý): {attn.shape} -> Chuẩn là [{BATCH_SIZE}, 8]")
