# News AI Project - Phan loai tin tuc da phuong thuc (Text + Image)

Du an xay dung he thong phan loai bai bao tieng Viet dua tren 2 loai du lieu:

- Van ban (tieu de + noi dung)
- Hinh anh minh hoa cua bai bao

Du an hien co 2 huong mo hinh:

- Baseline tu xay dung: CNN (anh) + BiLSTM-Attention (text)
- Transformer fine-tuning: ResNet50 + PhoBERT

## 1. Muc tieu du an

- Thu thap du lieu tin tuc tieng Viet tu RSS va trang bao
- Tao dataset da phuong thuc gom cap JSON + JPG
- Huan luyen mo hinh phan loai chu de tin tuc
- Danh gia tren tap Test tach rieng (train/val/test theo stratified split)

## 2. Cau truc thu muc

```text
news_ai_project/
|-- data/
|   |-- raw/
|   |   |-- texts/           # JSON moi bai bao
|   |   `-- images/          # Anh JPG tuong ung
|   `-- processed/           # Du phong cho du lieu da xu ly
|-- models/                  # Noi luu checkpoint model
|-- notebooks/
|   `-- pipeline_train.ipynb
|-- src/
|   |-- data/
|   |   |-- scraper.py
|   |   |-- dataset.py
|   |   `-- dataset_transformer.py
|   |-- nlp/
|   |   `-- nlp_model.py
|   |-- vision/
|   |   `-- cnn_model.py
|   |-- pipeline/
|   |   |-- multimodal.py
|   |   |-- multimodal_transformer.py
|   |   |-- train.py
|   |   `-- train_transformer.py
|   `-- utils/
|       |-- preprocess.py
|       `-- __init__.py
|-- requirements.txt
`-- README.md
```

## 3. Du lieu dau vao

Moi mau du lieu gom:

- 1 file JSON trong `data/raw/texts`
- 1 file anh JPG cung id trong `data/raw/images`

Nhan (classes) duoc suy ra tu truong `category` trong du lieu thuc te.

## 4. Cai dat moi truong

Yeu cau:

- Python 3.10+ (khuyen nghi 3.10 hoac 3.11)
- Pip moi
- Co GPU CUDA neu muon train nhanh

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Cach chay du an

Tat ca lenh ben duoi chay tu thu muc goc du an (`news_ai_project`).

### 5.1. Thu thap du lieu (crawler)

```powershell
python src/data/scraper.py
```

### 5.2. Huan luyen baseline (CNN + BiLSTM-Attention)

```powershell
python src/pipeline/train.py
```

Pipeline baseline:

- Text: clean/tokenize/build vocab trong `src/utils/preprocess.py`
- Image: CNN custom trong `src/vision/cnn_model.py`
- Text encoder: BiLSTM + Global Attention trong `src/nlp/nlp_model.py`
- Fusion: concat dac trung image/text trong `src/pipeline/multimodal.py`
- Train: early stopping + reduce lr + luu checkpoint tot nhat

Checkpoint mac dinh:

- `models/multimodal_model.pth`

### 5.3. Huan luyen transformer (ResNet50 + PhoBERT)

```powershell
python src/pipeline/train_transformer.py
```

Pipeline transformer:

- Vision encoder: ResNet50 pretrained ImageNet
- NLP encoder: `vinai/phobert-base-v2`
- Dataset tokenization bang `AutoTokenizer`
- Fine-tuning end-to-end voi AdamW
- Early stopping va luu model tot nhat

Checkpoint mac dinh:

- `models/transformer_model.pth`

### 5.4. Chay notebook tren Kaggle

Neu may local khong du RAM/VRAM, ban co the chay file notebook tren Kaggle de tan dung GPU mien phi.

#### Buoc 1: Chuan bi du lieu upload

- Nen du lieu thanh 1 file zip (chi can giu dung cau truc):
  - data/raw/texts/
  - data/raw/images/
- Dat ten goi y: news_ai_data.zip

#### Buoc 2: Tao Kaggle Notebook

- Vao Kaggle -> Code -> New Notebook
- Bat Accelerator = GPU (thuong la Tesla T4)
- Them dataset da upload vao Notebook

#### Buoc 3: Mo notebook pipeline

- Upload file notebooks/pipeline_train.ipynb len Kaggle (hoac import tu GitHub)
- Khong can chinh sua code trong notebook neu dataset da gan dung cau truc

#### Buoc 4: Chay theo thu tu cell

- Kiem tra detect GPU:

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

- Bam Run All (hoac chay lan luot tung cell tu tren xuong duoi).
- Khi train xong, download checkpoint tu muc Output cua Kaggle Notebook.

#### Buoc 5: Meo toi uu neu hay bi out-of-memory

- Giam BATCH_SIZE (vi du 16 -> 8 -> 4)
- Giam MAX_SEQ_LEN (vi du 256 -> 192)
- Dung baseline truoc, sau do moi chay transformer
- Tat bot data augmentation nang neu can toc do nhanh hon

## 6. Tom tat cac module chinh

- `src/data/scraper.py`:
  - Thu thap bai viet va anh tu RSS/website
  - Tao file JSON + JPG theo id

- `src/data/dataset.py`:
  - Dataset cho baseline
  - Tra ve: image tensor, text tensor, text length, label id

- `src/data/dataset_transformer.py`:
  - Dataset cho transformer
  - Tra ve dict gom image, input_ids, attention_mask, label

- `src/utils/preprocess.py`:
  - TextProcessor (clean text, tokenize, build vocab, pad/truncate)

- `src/nlp/nlp_model.py`:
  - NewsTextExtractor (Embedding -> BiLSTM -> Attention)

- `src/vision/cnn_model.py`:
  - NewsImageExtractor (CNN custom 4 block)

- `src/pipeline/multimodal.py`:
  - Hop nhat dac trung baseline va classifier

- `src/pipeline/multimodal_transformer.py`:
  - Hop nhat dac trung ResNet50 va PhoBERT

- `src/pipeline/train.py`:
  - Train/evaluate baseline, chia train/val/test, luu checkpoint

- `src/pipeline/train_transformer.py`:
  - Train/evaluate transformer, luu checkpoint

## 7. Logging va theo doi ket qua

Trong qua trinh train, he thong in:

- Train Accuracy theo epoch
- Validation Accuracy/Loss theo epoch
- Thong bao luu model tot nhat
- Ket qua cuoi cung tren tap Test

Ban co the redirect log ra file:

```powershell
python src/pipeline/train.py *> train_baseline.log
python src/pipeline/train_transformer.py *> train_transformer.log
```
