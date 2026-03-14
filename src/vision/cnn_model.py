import torch
import torch.nn as nn


class NewsImageExtractor(nn.Module):
    """
    Module Vision tự xây dựng (Custom CNN từ đầu).
    Nhiệm vụ: Nhận ảnh đầu vào và trích xuất ra một vector đặc trưng 1 chiều.
    """

    def __init__(self, output_dim=512, dropout=0.3):
        super(NewsImageExtractor, self).__init__()

        self.output_dim = output_dim

        # BLOCK 1: Nhìn chi tiết bề mặt (Cạnh, màu sắc)
        # Input: [Batch, 3, 224, 224] -> Output: [Batch, 32, 112, 112]
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(32),  # Chuẩn hóa giúp train nhanh và ổn định hơn
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Giảm nửa kích thước ảnh
        )

        # BLOCK 2: Nhìn hình khối cơ bản
        # Input: [Batch, 32, 112, 112] -> Output: [Batch, 64, 56, 56]
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # BLOCK 3: Nhìn họa tiết phức tạp
        # Input: [Batch, 64, 56, 56] -> Output: [Batch, 128, 28, 28]
        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # BLOCK 4: Nhìn tổng thể khái niệm (Đặc trưng mức cao)
        # Input: [Batch, 128, 28, 28] -> Output: [Batch, 256, 14, 14]
        self.block4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # FUSION PREP: Chuyển đổi ma trận 2D thành Vector 1D
        # AdaptiveAvgPool2d(1) sẽ ép mọi kích thước (dù là 14x14 hay 20x20) về đúng 1x1.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output: [Batch, 256, 1, 1]

        # Lớp Linear cuối cùng để ép chiều vector về đúng với output_dim mong muốn (VD: 512)
        self.fc_projection = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(256, output_dim), nn.ReLU()
        )

    def forward(self, images):
        """
        Luồng đi của dữ liệu hình ảnh qua các block CNN
        """
        # Trích xuất đặc trưng qua 4 lớp chập
        x = self.block1(images)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Gộp cụm toàn cục
        x = self.global_pool(x)  # Shape: [Batch, 256, 1, 1]

        # Dát mỏng Tensor 4D thành 2D
        x = torch.flatten(x, 1)  # Shape: [Batch, 256]

        # Chiếu lên không gian output_dim
        features = self.fc_projection(x)  # Shape: [Batch, output_dim]

        return features


if __name__ == "__main__":
    # Giả lập 1 mẻ gồm 4 bức ảnh RGB, kích thước 224x224
    dummy_images = torch.randn(4, 3, 224, 224)

    # Khởi tạo mô hình yêu cầu output ra vector 512 chiều
    model = NewsImageExtractor(output_dim=512)

    # Cho ảnh chạy qua mô hình
    output_features = model(dummy_images)

    print(f"Shape Input (Ảnh)       : {dummy_images.shape}")
    print(f"Shape Output (Đặc trưng): {output_features.shape}")

    # Xác nhận shape
    assert output_features.shape == (
        4,
        512,
    ), "Lỗi: Output shape không khớp với kỳ vọng!"
