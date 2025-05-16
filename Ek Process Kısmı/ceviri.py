import pandas as pd
from google.cloud import translate_v2 as translate
import os

# JSON key dosyasını belirt
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "fleet-reserve-459720-b5-4b23af070f8f.json"

# API istemcisini başlat
client = translate.Client()

# CSV oku
df = pd.read_csv('Skin_text_classifier.csv')
df = df.rename(columns={'Disease name': 'hastalik', 'Text': 'aciklama'})

# Çeviri fonksiyonu
def translate_text(text):
    try:
        result = client.translate(text, target_language='tr', format_='text')
        return result['translatedText']
    except Exception as e:
        print(f"❌ Hata: {e}")
        return text

# Hastalık kolonunu çevir
df['hastalik'] = df['hastalik'].apply(lambda x: translate_text(str(x)))
# Açıklama kolonunu çevir
df['aciklama'] = df['aciklama'].apply(lambda x: translate_text(str(x)))

# Kaydet
df[['hastalik', 'aciklama']].to_csv('hastalik_aciklama_google_tr.csv', index=False, encoding='utf-8-sig')

print("🎉 Google Translate API ile tüm CSV Türkçeye çevrildi.")
