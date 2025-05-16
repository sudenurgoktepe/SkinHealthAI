import pandas as pd
import google.generativeai as genai

# === AYARLAR ===
CSV_INPUT = "Skin_text_classifier_son.csv"
CSV_OUTPUT = "Skin_text_classifier_augmented.csv"
NUM_VARIATIONS = 30

API_KEY = "AIzaSyBkaffB-LrFC238e88ge6ZyECBYzmuM-Q8"

# === Gemini API Ayarla ===
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('models/gemini-1.5-pro')

def generate_variations(text, num_variations):
    try:
        prompt = (
            f"Aşağıdaki cilt açıklamasını Türkçe'de {num_variations} farklı doğal, kullanıcı dostu şekilde yeniden ifade et:\n\n"
            f"\"{text}\"\n\n"
            f"Sadece cümleleri sıralı olarak listele."
        )
        response = model.generate_content([{"role": "user", "parts": [prompt]}])
        content = response.candidates[0].content.parts[0].text
        variations = [line.strip() for line in content.split("\n") if line.strip()]
        return variations[:num_variations]
    except Exception as e:
        print(f"Gemini API hatası: {e}")
        return []

def main():
    print(f"CSV dosyası yükleniyor: {CSV_INPUT}")
    df = pd.read_csv(CSV_INPUT)
    print(f"Toplam satır: {len(df)}")

    augmented_rows = []
    for idx, row in df.iterrows():
        original_prompt = row['aciklama']
        print(f"[{idx + 1}/{len(df)}] Prompt işleniyor: {original_prompt[:50]}...")

        variations = generate_variations(original_prompt, NUM_VARIATIONS)
        if not variations:
            print(f"HATA: {original_prompt[:50]}... için varyasyon döndürmedi!")
            continue

        for v in variations:
            new_row = row.copy()
            new_row['aciklama'] = v
            augmented_rows.append(new_row)

    # Sonuçları birleştir & kaydet
    print(f"Varyasyonlar eklendi, toplam yeni satır: {len(augmented_rows)}")
    df_augmented = pd.DataFrame(augmented_rows)
    df_full = pd.concat([df, df_augmented], ignore_index=True)
    df_full.to_csv(CSV_OUTPUT, index=False)
    print(f"İşlem tamam. Çıktı dosyası: {CSV_OUTPUT}")

if __name__ == "__main__":
    main()
