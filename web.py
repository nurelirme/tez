import streamlit as st
import pandas as pd
import tempfile
import os
import numpy as np
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GProp import GProp_GProps
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from pathlib import Path


# Streamlit ayarları
st.set_page_config(page_title="STEP Analizi & Süre Tahmin", layout="wide")
st.markdown("""
<div style='background-color:#002147; padding:20px; border-radius:10px;'>
    <h1 style='text-align: center; color: white;'>
        🚀 Sampa Kalıp Üretiminde Termin Süresi Tahmini Projesi
    </h1>

</div>

<hr style='border: 1px solid white;'>

""", unsafe_allow_html=True)

st.markdown("""
Bu uygulama iki ana adımda çalışır:
1. **Model Eğitimi:** Daha önce tamamlanmış projelerin bulunduğu Excel dosyasını yükleyerek üretim süresi tahmin modeli oluşturulur.
2. **Yeni Proje Tahmini:** STEP dosyasını yükleyerek yeni bir proje için süre tahminleri elde edilir.
""")

# Sidebar: Model Eğitimi için Excel dosyası yükleme
st.sidebar.header("📂 Model Eğitimi")
st.sidebar.markdown("Tamamlanmış projeleri içeren Excel dosyasını buradan yükleyin.")
uploaded_excel = st.sidebar.file_uploader("📊 Excel dosyası (.xlsx)", type=["xlsx"], key="excel_uploader")

# Ana sayfa: STEP dosyası yükleme
st.header("📄 Yeni Proje için STEP Yükle")
st.markdown("Tahmin etmek istediğiniz yeni projenin STEP dosyasını yükleyin.")
uploaded_step = st.file_uploader("📁 STEP dosyası (.stp, .step)", type=["stp", "step"], key="step_uploader")


step_file_path = Path(__file__).resolve().parent / "ornek_dosya.stp"
if step_file_path.exists():
    with open(step_file_path, "rb") as f:
        st.sidebar.download_button(
            label="📥 Örnek STEP Dosyasını İndir",
            data=f,
            file_name="ornek_dosya.step",
            mime="application/octet-stream"
        )
else:
    st.sidebar.warning("📂 ornek_dosya.stp bulunamadı.")




excel_file_path = Path(__file__).resolve().parent / "ornek_excel.xlsx"

if excel_file_path.exists():
    with open(excel_file_path, "rb") as f:
        st.sidebar.download_button(
            label="📥 Örnek Excel Dosyasını İndir",
            data=f,
            file_name="ornek_excel.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.sidebar.warning("📂 ornek_excel.xlsx bulunamadı.")



# STEP analiz fonksiyonu
def analyze_step_file(file_path):
    reader = STEPControl_Reader()
    status = reader.ReadFile(file_path)
    if status != IFSelect_RetDone:
        return {"Hata": "STEP dosyası okunamadı."}

    reader.TransferRoots()
    shape = reader.Shape()

    # Yüzey sayısı
    face_exp = TopExp_Explorer(shape, TopAbs_FACE)
    face_count = 0
    while face_exp.More():
        face_count += 1
        face_exp.Next()

    # Solid (göz) sayısı
    solid_exp = TopExp_Explorer(shape, TopAbs_SOLID)
    solid_count = 0
    while solid_exp.More():
        solid_count += 1
        solid_exp.Next()

    # Hacim hesapla
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    volume = props.Mass()

    # Bounding box boyutları
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    dims = [xmax - xmin, ymax - ymin, zmax - zmin]
    min_dim, max_dim = min(dims), max(dims)
    thickness = min_dim
    aspect_ratio = max_dim / min_dim if min_dim > 0 else np.inf

    # Form derecesi (yaklaşık karmaşıklık)
    form_derecesi = min((face_count // 20) + 1, 5)

    if thickness >= 3.5 and volume > 50000:
        kalip_turu = "Plastik"
    elif thickness < 3.5 and volume <= 50000:
        kalip_turu = "Sac"
    else:
        if face_count < 250 and solid_count <= 3 and aspect_ratio > 8:
            kalip_turu = "Plastik"
        else:
            kalip_turu = "Sac"

        # Göz adedi tahmini
    if solid_count > 1:
        goz_adedi = solid_count
        goz_adedi_yorum = "🔍 Göz adedi solid sayısına göre otomatik belirlendi."
    elif volume > 0 and kalip_turu == "Plastik":
        goz_adedi = max(1, int(volume / 50000))
        goz_adedi_yorum = "ℹ️ Tek solid olduğu için hacme göre yaklaşık göz adedi tahmini yapıldı."
    else:
        goz_adedi = 1
        goz_adedi_yorum = "⚠️ Otomatik tespit mümkün değil, lütfen göz adedini manuel girin."

        # Analiz Zorluk Derecesi - Yüzey sayısına bağlı olarak
    if face_count < 100:
        analiz_zorluk = 1  # Kolay
    elif face_count < 500:
        analiz_zorluk = 3  # Orta
    else:
        analiz_zorluk = 5  # Zor

    result = {
        "Kalıp Türü": kalip_turu,
        "Form Derecesi": form_derecesi,
        "Yüzey Sayısı": face_count,
        "Hacim (mm³)": round(volume, 2),
        "Bounding Box Boyutları (mm)": [round(d, 2) for d in dims],
        "Tahmini Kalınlık (mm)": round(thickness, 2),
        "En boy oranı": round(aspect_ratio, 2),
        "Analiz Zorluk Derecesi": analiz_zorluk,  # Otomatik belirlenen analiz zorluk derecesi

    }

    # Ek öznitelikler
    if kalip_turu == "Plastik":
        result["Göz Adedi"] = max(1, solid_count)
        result["Göz Adedi Notu"] = goz_adedi_yorum

    else:
        result["Sac Kalınlığı"] = round(thickness, 2)
        result["Kalıp Adedi"] = max(1, int(volume / 50000))

    return result


# Model eğitimi
def encode_and_train(df, features, targets):
    df_encoded = df.copy()
    df_encoded.fillna(0, inplace=True)
    encoders = {}

    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            encoders[col] = LabelEncoder()
            df_encoded[col] = encoders[col].fit_transform(df_encoded[col].astype(str))
    X = df_encoded[features]
    models = {}
    for target in targets:
        y = df_encoded[target]
        model = RandomForestRegressor(n_estimators=100, random_state=43)
        model.fit(X, y)
        models[target] = model
    return models, encoders


def predict_with_model(models, input_data):
    predictions = {}
    for key, model in models.items():
        prediction = model.predict([input_data])[0]
        predictions[key] = round(prediction, 2)
    return predictions


def calculate_r2_score(models, df_encoded, feature_cols, target_cols):
    X = df_encoded[feature_cols]
    total_actual = df_encoded[target_cols].sum(axis=1)
    total_predicted = np.zeros(len(df_encoded))
    for target in target_cols:
        model = models[target]
        total_predicted += model.predict(X)
    return r2_score(total_actual, total_predicted)


def show_prediction_results(preds):
    st.success("🎯 Tahmin Başarılı")
    st.write("### ⏱ Süre Tahminleri")
    for k, v in preds.items():
        st.markdown(f"**{k}:** {v} saat")
    toplam = sum(preds.values())
    st.markdown(f"### 🧮 **Toplam Tahmini Termin Süresi:** {toplam} saat")

    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        preds.values(),
        labels=preds.keys(),
        autopct='%1.1f%%',
        startangle=140,
        textprops=dict(color="black", fontsize=10)
    )
    ax.axis('equal')
    plt.tight_layout()
    st.pyplot(fig)


def calculate_performance_metrics(models, df_encoded, feature_cols, target_cols):
    X = df_encoded[feature_cols]
    y_true_total = df_encoded[target_cols].sum(axis=1)

    y_pred_total = np.zeros(len(df_encoded))
    for target in target_cols:
        y_pred_total += models[target].predict(X)

    r2 = r2_score(y_true_total, y_pred_total)
    mae = mean_absolute_error(y_true_total, y_pred_total)
    mse = mean_squared_error(y_true_total, y_pred_total)
    rmse = np.sqrt(mse)
    # MAPE (Ortalama Mutlak Yüzde Hata) - küçük gerçek değerler için dikkatli kullan
    mape = np.mean(np.abs((y_true_total - y_pred_total) / np.where(y_true_total == 0, 1, y_true_total))) * 100

    return {
        "R²": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape
    }


if uploaded_excel:
    st.sidebar.success("Excel başarıyla yüklendi! Model oluşturuluyor...")
    df = pd.read_excel(uploaded_excel)
    df.fillna(0, inplace=True)
    feature_cols = ['Kalıp Türü', 'Form Derecesi', 'Yüzey Sayısı', 'Hacim', 'Göz Adedi',
                    'Sac Kalınlığı', 'Kalıp Adedi', 'Yolluk Tipi', 'Sac Cinsi', 'Hammadde Cinsi',
                    'Parça Toleransı', 'Analiz Zorluk Derecesi']
    feature_cols = [col for col in feature_cols if col in df.columns]
    target_cols = ['Tasarım Süresi', 'CAM Süresi', 'Talaşlı İmalat Süresi', 'Montaj Süresi']

    models, encoders = encode_and_train(df, feature_cols, target_cols)

    df_encoded = df.copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            encoder = LabelEncoder()
            df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))


    r2 = calculate_r2_score(models, df_encoded, feature_cols, target_cols)
    st.sidebar.success(f"📊 Modelin Toplam Termin Süresi Üzerinden R² Skoru: **{r2:.2f}**")
    st.sidebar.caption("R² skoru modelin doğruluk seviyesini gösterir. 1.00 mükemmel uyumdur.")
    # Performans metriklerini hesapla
    metrics = calculate_performance_metrics(models, df_encoded, feature_cols, target_cols)

    st.sidebar.markdown("### 📊 Model Performans Metrikleri")
    st.sidebar.write(f"**Determinasyon Katsayısı (R²):** {metrics['R²']:.3f}")
    st.sidebar.write(f"**Ortalama Mutlak Hata (MAE):** {metrics['MAE']:.3f}")
    st.sidebar.write(f"**Ortalama Kare Hata (MSE):** {metrics['MSE']:.3f}")
    st.sidebar.write(f"**Karekök Ortalama Kare Hata (RMSE):** {metrics['RMSE']:.3f}")
    st.sidebar.write(f"**Ortalama Mutlak Yüzde Hata (MAPE):** %{metrics['MAPE']:.2f}")

    # STEP dosyası yüklenmişse analiz ve tahmin işlemlerine başla
    if uploaded_step:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".stp") as tmp:
            tmp.write(uploaded_step.read())
            tmp_path = tmp.name

        st.subheader("📊 STEP Analiz Sonuçları")
        step_data = analyze_step_file(tmp_path)
        os.remove(tmp_path)

        if "Hata" in step_data:
            st.error(step_data["Hata"])
        else:
            cols = st.columns(2)
            for i, (k, v) in enumerate(step_data.items()):
                with cols[i % 2]:
                    st.markdown(f"""
                            <div style="
                            background-color:#f0f0f0;
                            color:#000000;
                            padding:12px;
                            border-radius:10px;
                            margin-bottom:10px;
                            border-left: 5px solid #ab47bc;
                            box-shadow: 1px 1px 4px rgba(0,0,0,0.05);
                            ">
                                <strong>{k}</strong><br>
                                {v}
                            </div>
                            """, unsafe_allow_html=True)

            st.subheader("🔍 Ek Bilgiler")

            form_derecesi_step = step_data.get("Form Derecesi", 0.0)
            yuzey_sayisi_step = step_data.get("Yüzey Sayısı", 0)
            hacim_step = step_data.get("Hacim (mm³)", 0.0)

            with st.expander("STEP'ten çekilemeyen verileri giriniz"):
                st.subheader("🛠️ Sonucu manuel olarak düzeltmek ister misiniz?")
                override = st.selectbox(
                    "Kalıp türü doğru mu?",
                    ["Evet, doğru", "Hayır - Bu Sac", "Hayır - Bu Plastik"],
                    key="override_select"
                )
                if override == "Hayır - Bu Sac":
                    step_data = {}  # Otomatik verileri sıfırla
                    step_data["Kalıp Türü"] = "Sac"
                elif override == "Hayır - Bu Plastik":
                    step_data = {}  # Otomatik verileri sıfırla
                    step_data["Kalıp Türü"] = "Plastik"

                step_data["Parça Toleransı"] = st.text_input("Parça Toleransı (örnek: ±0.01)", key="tolerans")
                step_data["Analiz Zorluk Derecesi"] = st.number_input("Analiz Zorluk Derecesi (1-5)", min_value=1,
                                                                      max_value=5, step=1, key="zorluk")
                # Bu kısım STEP analizinden gelen verilerle ön dolu
                step_data["Form Derecesi"] = st.number_input(
                    "Form Derecesi",
                    min_value=0.0,
                    step=0.1,
                    value=float(step_data.get("Form Derecesi", form_derecesi_step)),
                    key="form_derecesi"
                )

                step_data["Yüzey Sayısı"] = st.number_input(
                    "Yüzey Sayısı",
                    min_value=0,
                    step=1,
                    value=int(step_data.get("Yüzey Sayısı", yuzey_sayisi_step)),
                    key="yuzey_sayisi"
                )

                step_data["Hacim (mm³)"] = st.number_input(
                    "Hacim (mm³)",
                    min_value=0.0,
                    step=100.0,
                    value=float(step_data.get("Hacim (mm³)", hacim_step)),
                    key="hacim"
                )
                if step_data["Kalıp Türü"] == "Plastik":
                    step_data["Yolluk Tipi"] = st.text_input("Yolluk Tipi (örnek: Sıcak, Soğuk)", key="yolluk_tipi")
                    step_data["Göz Adedi"] = st.number_input("Göz Adedi", min_value=1, key="goz_adedi")
                    step_data["Hammadde Cinsi"] = st.text_input("Hammadde Cinsi (örnek: PP, ABS, vb.)",
                                                                key="hammadde_plastik")
                    step_data["Kalıp Adedi"] = st.number_input("Kalıp Adedi", min_value=1, key="kalip_adedi_plastik")
                    step_data["Operasyon Sayısı"] = st.number_input("Operasyon Sayısı", min_value=1, step=1,
                                                                    key="operasyon_sayisi_plastik")
                else:
                    step_data["Sac Kalınlığı"] = st.number_input("Sac Kalınlığı (mm)", min_value=0.1, step=0.1,
                                                                 key="sac_kalinlik")
                    step_data["Kalıp Adedi"] = st.number_input("Kalıp Adedi", min_value=1, key="kalip_adedi")
                    step_data["Sac Cinsi"] = st.text_input("Sac Cinsi (örnek: DKP, Paslanmaz)", key="sac_cinsi")
                    step_data["Operasyon Cinsi"] = st.text_input("Operasyon Cinsi (virgülle ayırın: Delme, Bükme, ...)",
                                                                 key="operasyon_cinsi")
                    step_data["Operasyon Sayısı"] = st.number_input("Operasyon Sayısı", min_value=1, step=1,
                                                                    key="operasyon_sayisi_sac")
                    step_data["Hammadde Cinsi"] = st.text_input("Hammadde Cinsi (örnek: St37, Alüminyum)",
                                                                key="hammadde_sac")

            if st.button("📊 STEP'e Göre Tahmin Et", key="predict_step_btn"):
                for col in feature_cols:
                    if col not in step_data:
                        step_data[col] = 0

                input_encoded = []
                for col in feature_cols:
                    val = step_data[col]
                    if col in encoders:
                        val = encoders[col].transform([val])[0] if val in encoders[col].classes_ else 0
                    input_encoded.append(val)

                preds = predict_with_model(models, input_encoded)
                show_prediction_results(preds)

    st.markdown("---")
