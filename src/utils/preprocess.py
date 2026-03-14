import re
from collections import Counter
import torch


class TextProcessor:
    """
    Class chuẩn kỹ sư để xử lý văn bản: Clean text, Tokenize, Build Vocab và Padding.
    """

    def __init__(self, max_vocab_size=20000, max_seq_len=256):
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len

        # Các token đặc biệt bắt buộc phải có trong NLP
        self.PAD_TOKEN = "<PAD>"  # Padding (bù số 0 cho câu ngắn)
        self.UNK_TOKEN = "<UNK>"  # Unknown (từ chưa từng gặp trong lúc train)

        # Từ điển Map (Ánh xạ)
        self.word2idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        self.vocab_built = False

    def clean_text(self, text):
        """Làm sạch văn bản chuẩn NLP: Hỗ trợ đa ngôn ngữ và xử lý dấu câu"""
        text = str(text).lower()

        # 1. Xóa các thẻ HTML (nếu còn sót)
        text = re.sub(r"<[^>]+>", " ", text)

        # 2. GIỮ LẠI MỌI NGÔN NGỮ (Kỹ thuật Unicode)
        # \w: Giữ lại tất cả chữ cái của mọi ngôn ngữ (Anh, Việt, Pháp...) và số.
        # \s: Giữ lại khoảng trắng.
        # .,!?: Giữ lại các dấu câu cơ bản để mô hình hiểu ngữ cảnh.
        text = re.sub(r"[^\w\s.,!?]", " ", text)

        # 3. KỸ THUẬT NLP NÂNG CAO: Tách dấu câu khỏi chữ
        text = re.sub(r"([.,!?])", r" \1 ", text)

        # 4. Xóa các khoảng trắng thừa
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize(self, text):
        """Tách câu thành các từ đơn (Khoảng trắng)"""
        return self.clean_text(text).split()

    def build_vocab(self, list_of_texts):
        """
        Xây dựng từ điển từ tập dữ liệu Train.
        Hàm này đếm tần suất xuất hiện của các từ và chỉ giữ lại những từ phổ biến nhất.
        """
        print("Đang xây dựng từ điển (Vocabulary)...")
        counter = Counter()

        # Duyệt qua từng bài báo để đếm từ
        for text in list_of_texts:
            tokens = self.tokenize(text)
            counter.update(tokens)

        # Chỉ lấy top 'max_vocab_size' từ xuất hiện nhiều nhất
        most_common = counter.most_common(self.max_vocab_size)

        # Cập nhật vào Dictionary (Bắt đầu từ index 2 vì 0 và 1 đã dành cho PAD và UNK)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self.vocab_built = True
        print(f"Xây dựng xong từ điển với {len(self.word2idx)} từ vựng.")

    def text_to_tensor(self, text):
        """
        Biến 1 đoạn text (String) thành Tensor gồm các con số (Index).
        Tự động cắt ngắn (Truncate) nếu quá dài và đệm số 0 (Pad) nếu quá ngắn.
        Trả về Tensor với kích thước [seq_len] và độ dài thực tế của câu (Trước khi Pad).
        """
        if not self.vocab_built:
            raise ValueError("Phải gọi build_vocab() trước khi chuyển đổi text!")

        # 1. Tách từ
        tokens = self.tokenize(text)

        # 2. Chuyển chữ thành số. Nếu từ không có trong từ điển thì biến thành <UNK> (số 1)
        indices = [self.word2idx.get(word, 1) for word in tokens]

        # 3. Ghi nhận độ dài thực tế của câu (Rất quan trọng cho Pack Padded Sequence của LSTM)
        actual_length = min(len(indices), self.max_seq_len)
        if actual_length == 0:
            actual_length = 1  # Tránh lỗi PyTorch khi gặp câu rỗng

        # 4. Ép kích thước mảng về đúng max_seq_len (Ví dụ: 256)
        if len(indices) > self.max_seq_len:
            # Cắt ngắn (Truncate)
            indices = indices[: self.max_seq_len]
        else:
            # Bù số 0 (Padding) vào cuối mảng
            indices += [0] * (self.max_seq_len - len(indices))

        # 5. Đóng gói thành PyTorch Tensor
        return torch.tensor(indices, dtype=torch.long), torch.tensor(
            actual_length, dtype=torch.long
        )
