import os
import glob
import pickle
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.neighbors import KNeighborsClassifier
from icrawler.builtin import BaiduImageCrawler, BingImageCrawler, GoogleImageCrawler


# --------------------------------------------------
# 1. 多搜索引擎備用爬蟲 (確保下載超過 150 張)
# --------------------------------------------------
def fetch_images(category_name, keywords, target_count, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    crawlers = [BaiduImageCrawler, BingImageCrawler, GoogleImageCrawler]

    current_count = len([f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))])
    print(f"[{category_name}] 目前已有 {current_count} 張圖片...")

    for kw in keywords:
        for crawler_cls in crawlers:
            if current_count >= target_count:
                break
            needed = target_count - current_count + 10  # 多抓 10 張備用
            crawler_name = crawler_cls.__name__
            print(f"[{category_name}] 嘗試使用 {crawler_name} 搜尋關鍵字 '{kw}'...")

            try:
                crawler = crawler_cls(storage={'root_dir': save_dir}, log_level=40)
                crawler.crawl(keyword=kw, max_num=needed)
            except Exception as e:
                print(f"爬取失敗 ({crawler_name}): {e}")

            current_count = len([f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))])
            print(f"目前累積 {category_name} 圖片數: {current_count}/{target_count}")


def download_dataset():
    cat_kw = ['cat photo', 'cute cat', 'pet cat', 'cat face', 'kitten']
    dog_kw = ['dog photo', 'cute dog', 'pet dog', 'dog face', 'puppy']

    # 強制抓滿 160 張 (預留損壞清洗空間)
    fetch_images("Cat", cat_kw, 160, "dataset/cat")
    fetch_images("Dog", dog_kw, 160, "dataset/dog")

    # 下載 10 張 Unseen 測試圖 (5 貓 5 狗)
    fetch_images("Unseen Cat", ['cat close up'], 5, "dataset/unseen_cat")
    fetch_images("Unseen Dog", ['dog close up'], 5, "dataset/unseen_dog")


# --------------------------------------------------
# 2. ResNet50 特徵編碼器 (Encoding)
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = models.ResNet50_Weights.DEFAULT
resnet = models.resnet50(weights=weights)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_feature(img_path):
    """將圖片轉換成 2048 維特徵向量"""
    try:
        image = Image.open(img_path).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = resnet(tensor).squeeze().cpu().numpy()
        return feature / np.linalg.norm(feature)  # L2 Normalization
    except Exception:
        return None


# --------------------------------------------------
# 3. 整合處理與建立 KNN 模型
# --------------------------------------------------
def process():
    download_dataset()

    image_paths, features, labels = [], [], []

    print("\n正在進行 ResNet50 特徵編碼...")
    for path in glob.glob("dataset/cat/*"):
        feat = extract_feature(path)
        if feat is not None:
            features.append(feat)
            image_paths.append(path)
            labels.append(0)  # 0 代表貓

    for path in glob.glob("dataset/dog/*"):
        feat = extract_feature(path)
        if feat is not None:
            features.append(feat)
            image_paths.append(path)
            labels.append(1)  # 1 代表狗

    features = np.array(features)
    labels = np.array(labels)

    cat_cnt = sum(labels == 0)
    dog_cnt = sum(labels == 1)
    print(f"\n✅ 特徵提取完成！成功讀取 - 貓咪: {cat_cnt} 張, 狗狗: {dog_cnt} 張 (總數: {len(features)})")

    if cat_cnt < 150 or dog_cnt < 150:
        print("⚠️ 警告：圖片數量不足 150 張，請重新執行腳本嘗試補抓！")
        return

    # 建立 KNN 分類器 (使用 Cosine Similarity 距離)
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn.fit(features, labels)

    # 儲存資料庫與模型
    with open("cbir_knn_data.pkl", "wb") as f:
        pickle.dump({
            'features': features,
            'image_paths': image_paths,
            'labels': labels,
            'knn_model': knn
        }, f)
    print("💾 已成功保存特徵庫與 KNN 模型至 'cbir_knn_data.pkl'！")


if __name__ == "__main__":
    process()