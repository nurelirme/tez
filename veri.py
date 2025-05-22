import pandas as pd
import random

# Örnek değer havuzları
kalip_turleri = ["Plastik", "Sac"]
yolluk_tipleri = ["Sıcak", "Soğuk"]
sac_cinsleri = ["DKP", "Paslanmaz", "Galvanizli", "Alüminyum", "Bakır", "Titanyum"]
hammadde_plastik = ["PP", "ABS", "PA6", "POM", "TPU", "PP COP", "PA 66", "HDPE", "LDPE", "PC"]
hammadde_sac = ["St37", "StW12", "StW22", "X5CrNi18-10(304)", "S235JR", "DC01", "X2CrNi18-9", "X6Cr17"]
parca_toleranslari = ["0.018", "0.029", "0.047", "0.062", "0.072", "0.08", "0.1", "0.02", "0.05"]
operasyon_cinsleri = ["delme", "bükme", "prograsif", "kesme", "form", "lazer", "ekstrüzyon", "enjeksiyon", "pres"]

# Veri üretimi
data = []

for _ in range(5000):
    kalip_turu = random.choice(kalip_turleri)
    form_derecesi = random.randint(1, 5)
    yuzey_sayisi = random.randint(50, 800)
    hacim = round(random.uniform(50, 900), 2)
    analiz_zorluk = min(5, max(1, yuzey_sayisi // 150))

    if kalip_turu == "Plastik":
        goz_adedi = random.randint(1, 12)
        sac_kalinligi = 0
        kalip_adedi = 1  # Plastik için sabit
        yolluk_tipi = random.choice(yolluk_tipleri)
        sac_cinsi = ""
        hammadde_cinsi = random.choice(hammadde_plastik)
        operasyon_cinsi = random.choice(["enjeksiyon", "ekstrüzyon"])
    else:
        goz_adedi = 0
        sac_kalinligi = round(random.uniform(1.5, 5.0), 2)
        kalip_adedi = random.randint(1, 3)
        yolluk_tipi = ""
        sac_cinsi = random.choice(sac_cinsleri)
        hammadde_cinsi = random.choice(hammadde_sac)
        operasyon_cinsi = ", ".join(random.sample(operasyon_cinsleri, random.randint(1, 3)))

    parca_toleransi = random.choice(parca_toleranslari)
    operasyon_sayisi = operasyon_cinsi.count(",") + 1 if operasyon_cinsi else 0

    # Süreler
    tasarim_suresi = random.randint(4, 50)
    cam_suresi = random.randint(20, 80)

    # ✅ Güncellenmiş talaşlı imalat süresi hesabı
    if kalip_turu == "Plastik":
        imalat_suresi = random.randint(120, 240)
    else:
        imalat_suresi = random.randint(200, 320)
        if "prograsif" in operasyon_cinsi:
            imalat_suresi += 80
        if "bükme" in operasyon_cinsi:
            imalat_suresi += 40
        if "delme" in operasyon_cinsi:
            imalat_suresi += 30
        if operasyon_sayisi > 2:
            imalat_suresi += (operasyon_sayisi - 2) * 25
        imalat_suresi += int(kalip_adedi * 15)
        imalat_suresi += int((sac_kalinligi - 1.5) * 20)

    montaj_suresi = random.randint(9, 45)

    row = {
        "Proje Adı": f"PROJE-{random.randint(1, 300)}",
        "Kalıp Türü": kalip_turu,
        "Form Derecesi": form_derecesi,
        "Yüzey Sayısı": yuzey_sayisi,
        "Hacim": hacim,
        "Göz Adedi": goz_adedi,
        "Yolluk Tipi": yolluk_tipi,
        "Kalıp Adedi": kalip_adedi,
        "Sac Kalınlığı": sac_kalinligi,
        "Sac Cinsi": sac_cinsi,
        "Operasyon Cinsi": operasyon_cinsi,
        "Analiz Zorluk Derecesi": analiz_zorluk,
        "Parça Toleransı": parca_toleransi,
        "Hammadde Cinsi": hammadde_cinsi,
        "Operasyon Sayısı": operasyon_sayisi,
        "Tasarım Süresi": tasarim_suresi,
        "CAM Süresi": cam_suresi,
        "Talaşlı İmalat Süresi": imalat_suresi,
        "Montaj Süresi": montaj_suresi
    }

    data.append(row)

# DataFrame'e çevir ve Excel'e yaz
df = pd.DataFrame(data)
df.to_excel("ornek_excel.xlsx", index=False)

print("✅ Gerçekçi sürelerle güncellenmiş veri dosyası oluşturuldu.")



model_path = "model.pkl"
model = None

if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.success("Eğitimli model yüklendi.STEP dossayı yükleyip tahmin yapın. ")
else:
    st.warning("Model dosyası bulunamadı. Lütfen veri yükleyip eğitin.")
# Kaydet
joblib.dump(model, "model.pkl")
