import streamlit as st
import pickle
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
        return pickle.load(f)


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
with tab2:
    st.subheader("10 張未見圖片測試 (Performance on 10 Unseen Images)")

    if st.button("開始評估 10 張未見圖片"):
        cats = glob.glob("dataset/unseen_cat/*")[:5]
        dogs = glob.glob("dataset/unseen_dog/*")[:5]
        test_samples = [(p, 0) for p in cats] + [(p, 1) for p in dogs]

        failed_cases = []

        st.write("### 測試矩陣：")
        cols = st.columns(5)
        for i, (path, true_label) in enumerate(test_samples):
            img = Image.open(path)
            feat = extract_single_feature(img)
            pred_label = db['knn_model'].predict([feat])[0]

            correct = (pred_label == true_label)
            if not correct:
                failed_cases.append((path, true_label, pred_label))

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

        # 失敗案例展示 (Failed Cases)
        st.divider()
        st.subheader("⚠️ 失敗案例視覺化與分析 (Failed Cases)")
        if len(failed_cases) == 0:
            st.balloons()
            st.success("🎉 完美！10 張未見圖片全部分類正確！")
        else:
            f_cols = st.columns(len(failed_cases))
            for i, (path, t_l, p_l) in enumerate(failed_cases):
                with f_cols[i]:
                    st.image(path, width="stretch")
                    st.caption(f"真實: {'貓' if t_l == 0 else '狗'} ➔ 錯判為: {'貓' if p_l == 0 else '狗'}")

            st.markdown("""
            **失敗原因分析 (Failure Analysis)：**
            1. **背景佔比過高**：背景（如草地、床單）佔用過多特徵空間。
            2. **視角/特寫特殊**：極端特寫（如只照到鼻子）會丟失耳形與臉型特徵。
            3. **特徵空間重疊**：部分長毛小型犬特徵與貓咪較為接近。
            """)