import streamlit as st
import pickle
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity
import glob

st.set_page_config(page_title="貓狗 CBIR 與 KNN 系統", layout="wide")


# --------------------------------------------------
# 載入模型與資料庫
# --------------------------------------------------
@st.cache_resource
def load_resnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.ResNet50_Weights.DEFAULT
    resnet = models.resnet50(weights=weights)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
    resnet.eval()
    return resnet, device


@st.cache_data
def load_db():
    with open("cbir_knn_data.pkl", "rb") as f:
        db = pickle.load(f)
    keep = [i for i, p in enumerate(db['image_paths']) if os.path.exists(p)]
    if len(keep) < len(db['image_paths']):
        print(f"WARNING: filtering out {len(db['image_paths']) - len(keep)} missing image(s)")
    return {
        'features': db['features'][keep],
        'image_paths': [db['image_paths'][i] for i in keep],
        'labels': db['labels'][keep],
        'knn_model': db['knn_model'],
    }


resnet, device = load_resnet()
db = load_db()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_single_feature(image):
    tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = resnet(tensor).squeeze().cpu().numpy()
    return feat / np.linalg.norm(feat)


# --------------------------------------------------
# UI 介面
# --------------------------------------------------
st.title("🐱🐶 貓狗圖像檢索 (CBIR) & KNN 分類網站")
st.write(
    f"資料庫狀態：已載入 **{len(db['features'])}** 張圖片（貓: {sum(db['labels'] == 0)} 張, 狗: {sum(db['labels'] == 1)} 張）")

tab1, tab2 = st.tabs(["🔍 上傳查詢：相似檢索與 KNN 分類", "📊 10 張未見圖片測試與失敗案例"])

# --------------------------------------------------
# TAB 1: 整合 Task 2 & Task 3
# --------------------------------------------------
with tab1:
    c_sample, c_upload = st.columns([1, 3])
    with c_sample:
        if st.button("🎲 使用一張範例查詢圖片"):
            st.session_state['sample_query'] = Image.open(np.random.choice(sorted(glob.glob("dataset/*/*"))))
            if 'uploaded_query' in st.session_state:
                del st.session_state['uploaded_query']
    with c_upload:
        uploaded_file = st.file_uploader("選擇上傳一張貓或狗的圖片...", type=["jpg", "jpeg", "png"],
                                         key="query_uploader")

    if uploaded_file is not None:
        st.session_state['uploaded_query'] = Image.open(uploaded_file)
        if 'sample_query' in st.session_state:
            del st.session_state['sample_query']

    query_img = st.session_state.get('uploaded_query') or st.session_state.get('sample_query')

    if query_img is not None:
        is_sample = 'uploaded_query' not in st.session_state and 'sample_query' in st.session_state

        col_q, col_res = st.columns([1, 2])

        with col_q:
            st.image(query_img, caption="範例查詢圖片 (Sample)" if is_sample else "上傳的 Query Image",
                     width="stretch")
            query_feat = extract_single_feature(query_img)

            # KNN 分類預測
            pred = db['knn_model'].predict([query_feat])[0]
            proba = db['knn_model'].predict_proba([query_feat])[0]
            label_text = "🐱 貓 (Cat)" if pred == 0 else "🐶 狗 (Dog)"

            st.success(f"**KNN 分類結果**：{label_text}")
            st.write(f"Confidence (信心度)：貓 {proba[0] * 100:.1f}% | 狗 {proba[1] * 100:.1f}%")

        # 計算 Cosine Similarity
        sims = cosine_similarity([query_feat], db['features'])[0]
        sorted_indices = np.argsort(sims)[::-1]

        top5_sim = sorted_indices[:5]
        top5_diff = sorted_indices[-5:][::-1]

        st.divider()
        st.subheader("🔥 Top 5 最相似圖片 (Most Similar)")
        cols_sim = st.columns(5)
        for i, idx in enumerate(top5_sim):
            with cols_sim[i]:
                st.image(db['image_paths'][idx], width="stretch")
                st.caption(f"相似度: {sims[idx]:.4f}\n類別: {'貓' if db['labels'][idx] == 0 else '狗'}")

        st.subheader("❄️ Top 5 最不相似圖片 (Most Dissimilar)")
        cols_diff = st.columns(5)
        for i, idx in enumerate(top5_diff):
            with cols_diff[i]:
                st.image(db['image_paths'][idx], width="stretch")
                st.caption(f"相似度: {sims[idx]:.4f}\n類別: {'貓' if db['labels'][idx] == 0 else '狗'}")

# --------------------------------------------------
# TAB 2: Task 4 (10 Unseen Images Performance)
# --------------------------------------------------
def run_eval(samples):
    """samples: list of (image_source, true_label). source = path or file-like."""
    failed_cases = []

    st.write("### 測試矩陣：")
    cols = st.columns(5)
    for i, (src, true_label) in enumerate(samples):
        img = Image.open(src).convert('RGB')
        feat = extract_single_feature(img)
        pred_label = db['knn_model'].predict([feat])[0]

        correct = (pred_label == true_label)
        if not correct:
            failed_cases.append((img, true_label, pred_label))

        with cols[i % 5]:
            st.image(img, width="stretch")
            t_str = "貓" if true_label == 0 else "狗"
            p_str = "貓" if pred_label == 0 else "狗"
            if correct:
                st.success(f"真實:{t_str} | 預測:{p_str}")
            else:
                st.error(f"真實:{t_str} | 預測:{p_str}")
        if i == 4:
            st.divider()

    st.divider()
    st.subheader("⚠️ 失敗案例視覺化與分析 (Failed Cases)")
    if len(failed_cases) == 0:
        st.balloons()
        st.success("🎉 完美！全部分類正確！")
    else:
        f_cols = st.columns(len(failed_cases))
        for i, (img, t_l, p_l) in enumerate(failed_cases):
            with f_cols[i]:
                st.image(img, width="stretch")
                st.caption(f"真實: {'貓' if t_l == 0 else '狗'} ➔ 錯判為: {'貓' if p_l == 0 else '狗'}")

        st.markdown("""
        **失敗原因分析 (Failure Analysis)：**
        1. **背景佔比過高**：背景（如草地、床單）佔用過多特徵空間。
        2. **視角/特寫特殊**：極端特寫（如只照到鼻子）會丟失耳形與臉型特徵。
        3. **特徵空間重疊**：部分長毛小型犬特徵與貓咪較為接近。
        """)


with tab2:
    st.subheader("10 張未見圖片測試 (Performance on 10 Unseen Images)")

    mode = st.radio("選擇測試方式：", ["🎲 隨機抽取 (從 20 張未見圖片)", "📤 手動上傳 10 張照片"])

    if mode.startswith("🎲"):
        cats_pool = glob.glob("dataset/unseen_cat/*")
        dogs_pool = glob.glob("dataset/unseen_dog/*")
        st.caption(f"未見圖片池：貓 {len(cats_pool)} 張 + 狗 {len(dogs_pool)} 張 = {len(cats_pool) + len(dogs_pool)} 張")

        if st.button("🎲 隨機抽 10 張並開始評估"):
            rng = np.random.RandomState()
            cats = rng.choice(cats_pool, size=5, replace=False)
            dogs = rng.choice(dogs_pool, size=5, replace=False)
            test_samples = [(c, 0) for c in cats] + [(d, 1) for d in dogs]
            rng.shuffle(test_samples)
            run_eval(test_samples)

    else:
        st.caption("請手動上傳 10 張未見圖片 (真實類別由你指定)。")
        col_c, col_d = st.columns(2)
        with col_c:
            up_cats = st.file_uploader("🐱 上傳貓咪圖片 (可多選)", type=["jpg", "jpeg", "png"],
                                       accept_multiple_files=True, key="up_cats")
        with col_d:
            up_dogs = st.file_uploader("🐶 上傳狗狗圖片 (可多選)", type=["jpg", "jpeg", "png"],
                                       accept_multiple_files=True, key="up_dogs")

        if st.button("開始評估上傳圖片"):
            samples = [(f, 0) for f in up_cats] + [(f, 1) for f in up_dogs]
            samples = samples[:10]
            if len(samples) == 0:
                st.warning("請先上傳至少一張圖片！")
            else:
                st.caption(f"已選取 {len(samples)} 張圖片開始評估")
                run_eval(samples)