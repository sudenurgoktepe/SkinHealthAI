import pandas as pd

df = pd.read_csv("Skin_text_classifier_augmented.csv")

# Sadece 'aciklama' kolonundaki birebir duplicate'ları göster
duplicates = df[df.duplicated(subset=['aciklama'], keep=False)]

print(f"Toplam duplicate bulunan satır sayısı: {len(duplicates)}")
print(duplicates[['aciklama']].drop_duplicates().head(10))  # İlk 10 benzersiz duplicate'i göster
