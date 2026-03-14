import os
import json
import time
import requests
from bs4 import BeautifulSoup

# CẤU HÌNH HỆ THỐNG
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEXT_DIR = os.path.join(BASE_DIR, "data", "raw", "texts")
IMAGE_DIR = os.path.join(BASE_DIR, "data", "raw", "images")

os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# Giới hạn số lượng bài cào cho MỖI chuyên mục (Chỉnh to lên nếu muốn train model thật)
MAX_ARTICLES_PER_CATEGORY = 100

# Nguồn RSS của VnExpress (Đảm bảo tự động cập nhật tin mới nhất)
RSS_FEEDS = {
    "World": [
        "https://vnexpress.net/rss/the-gioi.rss",
        "https://tuoitre.vn/rss/the-gioi.rss",
        "https://thanhnien.vn/rss/the-gioi.rss",
    ],
    "Sports": [
        "https://vnexpress.net/rss/the-thao.rss",
        "https://tuoitre.vn/rss/the-thao.rss",
        "https://thanhnien.vn/rss/the-thao.rss",
    ],
    "Business": [
        "https://vnexpress.net/rss/kinh-doanh.rss",
        "https://tuoitre.vn/rss/kinh-doanh.rss",
        "https://thanhnien.vn/rss/kinh-doanh.rss",
    ],
    "Tech": [
        "https://vnexpress.net/rss/so-hoa.rss",
        "https://tuoitre.vn/rss/cong-nghe.rss",
        "https://thanhnien.vn/rss/cong-nghe-game.rss",
    ],
    "Entertainment": [
        "https://vnexpress.net/rss/giai-tri.rss",
        "https://tuoitre.vn/rss/giai-tri.rss",
        "https://thanhnien.vn/rss/giai-tri.rss",
    ],
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


# CÁC HÀM XỬ LÝ (PIPELINE)
def get_article_urls_from_rss(rss_url, max_urls):
    """Đọc RSS XML và trích xuất danh sách link bài báo"""
    urls = []
    try:
        response = requests.get(rss_url, headers=HEADERS, timeout=10)
        # Sử dụng trình phân tích XML (lưu ý: cần cài lxml hoặc dùng html.parser mặc định)
        soup = BeautifulSoup(
            response.content, "html.parser"
        )  # html.parser vẫn đọc được tag <item>

        items = soup.find_all("item")
        for item in items:
            link_tag = item.find("link")
            if link_tag and link_tag.next_sibling:
                # Trích xuất link thực sự từ trong thẻ CDATA hoặc text của thẻ link
                link = (
                    link_tag.next_sibling.strip()
                    if not link_tag.text
                    else link_tag.text
                )
                if (
                    "vnexpress.net" in link
                    and "podcast" not in link
                    and "video" not in link
                ):
                    urls.append(link)
            if len(urls) >= max_urls:
                break
    except Exception as e:
        print(f"Lỗi đọc RSS {rss_url}: {e}")
    return urls


def download_image(image_url, save_path):
    """Tải và lưu ảnh xuống ổ cứng"""
    try:
        response = requests.get(image_url, headers=HEADERS, stream=True, timeout=10)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
    except Exception as e:
        print(f"Lỗi tải ảnh {image_url}: {e}")
    return False


def scrape_and_save_article(url, category, article_id):
    """Cào chi tiết 1 bài báo và lưu thành file"""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")

        # 1. Tiêu đề (thẻ h1)
        title_tag = soup.find("h1", class_="title-detail")
        if not title_tag:
            return False
        title = title_tag.get_text().strip()

        # 2. Nội dung (Gom tất cả các thẻ p class Normal)
        paragraphs = soup.find_all("p", class_="Normal")
        if not paragraphs:
            return False
        # Chỉ lấy khoảng 10 đoạn đầu tiên để tránh file quá nặng
        content = " ".join([p.get_text().strip() for p in paragraphs[:10]])

        # 3. Ảnh (Tìm ảnh thumbnail đầu tiên trong bài)
        image_url = ""
        # Thường ảnh trong bài VnExpress nằm trong thẻ picture -> img
        img_tag = soup.find("img", itemprop="contentUrl")
        if not img_tag:
            meta_image = soup.find("meta", property="og:image")
            if meta_image:
                image_url = meta_image["content"]
        else:
            image_url = img_tag.get("data-src") or img_tag.get("src")

        if not image_url or "gif" in image_url.lower():
            return False

        # --- TIẾN HÀNH LƯU TRỮ ---
        file_prefix = f"{category}_{article_id:04d}"  # Thêm padding số 0, VD: Tech_0012

        # Lưu ảnh trước, nếu ảnh tải lỗi thì bỏ qua bài này luôn (đảm bảo Dataset sạch)
        image_path = os.path.join(IMAGE_DIR, f"{file_prefix}.jpg")
        if download_image(image_url, image_path):
            text_data = {
                "id": file_prefix,
                "url": url,
                "category": category,
                "title": title,
                "content": content,
                "image_file": f"{file_prefix}.jpg",
            }

            text_path = os.path.join(TEXT_DIR, f"{file_prefix}.json")
            with open(text_path, "w", encoding="utf-8") as f:
                json.dump(text_data, f, ensure_ascii=False, indent=4)

            return True

    except Exception as e:
        print(f"Lỗi xử lý bài báo {url}: {e}")
        pass

    return False


if __name__ == "__main__":
    total_saved = 0

    for category, rss_url in RSS_FEEDS.items():
        print(f"\n📡 Đang quét RSS chuyên mục: [{category}]")
        urls = get_article_urls_from_rss(
            rss_url, max_urls=MAX_ARTICLES_PER_CATEGORY + 20
        )  # Lấy dư ra phòng link lỗi

        print(f"Tìm thấy {len(urls)} links. Đang tiến hành tải nội dung...")

        saved_in_category = 0
        for url in urls:
            if saved_in_category >= MAX_ARTICLES_PER_CATEGORY:
                break  # Đủ KPI cho chuyên mục này thì dừng

            # Đánh index bài báo liên tục
            article_index = total_saved + 1

            if scrape_and_save_article(url, category, article_index):
                saved_in_category += 1
                total_saved += 1
                # In đếm tiến độ trên cùng 1 dòng
                print(
                    f"\r  -> Đã lưu: {saved_in_category}/{MAX_ARTICLES_PER_CATEGORY} bài",
                    end="",
                )

            time.sleep(0.5)  # Nghỉ nửa giây để không bị VnExpress chặn IP (Rate limit)

        print(f"\nHoàn thành chuyên mục [{category}].")

    print(f"TỔNG KẾT: Đã thu thập thành công {total_saved} bài báo chuẩn!")
    print(f"Dữ liệu lưu tại: {BASE_DIR}/data/raw/")
