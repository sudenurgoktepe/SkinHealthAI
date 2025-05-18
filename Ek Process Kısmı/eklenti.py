import pandas as pd


df = pd.read_csv('Skin_text_classifier_urun_commalarli.csv')
'''

unique_diseases = df['hastalik'].unique()

print(unique_diseases)
print(f"Toplam farklı hastalık sayısı: {len(unique_diseases)}")
'''

'''
# Hastalık -> Cilt Tipi eşleştirmesi
hastalik_cilt_mapping = {
    'Vitiligo': 'tum',
    'Uyuz': 'tum',
    'Kurdeşen (Ürtiker)': 'hassas',
    'Folikülit': 'yagli',
    'Egzama': 'kuru',
    'Saçkıran': 'tum',
    'Atlet Ayağı (Tinea Pedis)': 'yagli',
    'Gül hastalığı': 'hassas',
    'Sedef hastalığı': 'tum',
    'Zona (Herpes Zoster)': 'tum',
    'İmpetigo hastalığı': 'hassas',
    'Kontakt Dermatit': 'hassas',
    'Akne': 'yagli'
}

# Yeni keyword bazlı cilttipi kolonu ekle
df['cilttipi'] = df['hastalik'].map(hastalik_cilt_mapping)

# Kaydet
df.to_csv('Skin_text_classifier_cilttipli_keyword.csv', index=False, encoding='utf-8-sig')

print("🎯 Keyword bazlı cilttipi kolonun eklendi: Skin_text_classifier_cilttipli_keyword.csv")
'''
'''
hastalik_urun_mapping = {
    'Vitiligo': 'SPF50+ güneş koruyucu, Kamuflaj krem, Topikal kortikosteroid',
    'Uyuz': 'Permetrin %5 krem, İvermektin, Antihistaminik krem',
    'Kurdeşen (Ürtiker)': 'Antihistaminik krem, Seramid bariyer krem, Aloe vera jel',
    'Folikülit': 'Benzoyl Peroxide %5, Salisilik asitli temizleyici, Tea tree oil',
    'Egzama': 'Seramid nemlendirici, Topikal kortikosteroid krem, Lipid bariyer krem',
    'Saçkıran': 'Antifungal krem (Terbinafin), Antiseptik sabun, Çay ağacı yağı',
    'Atlet Ayağı (Tinea Pedis)': 'Antifungal sprey, Ayak deodorantı, Çinko pudra',
    'Gül hastalığı': 'Azelaik asit krem, Sükralfat bariyer krem, Termal sprey',
    'Sedef hastalığı': 'Katranlı şampuan, Salisilik asit krem, Nemlendirici',
    'Zona (Herpes Zoster)': 'Asiklovir krem, Yatıştırıcı losyon, Lidokainli krem',
    'İmpetigo hastalığı': 'Mupirosin merhem, Antiseptik temizleyici, Kurutucu pudra',
    'Kontakt Dermatit': 'Topikal kortikosteroid krem, Seramid bariyer krem, Termal sprey',
    'Akne': 'Benzoyl Peroxide %5-10, Salisilik Asit %2, Niacinamide serum, Retinoid krem'
}

# Yeni kolon ekle (comma ayrımlı)
df['onerilen_urun'] = df['hastalik'].map(hastalik_urun_mapping)

# Kaydet
df.to_csv('Skin_text_classifier_urun_commalarli.csv', index=False, encoding='utf-8-sig')

print("🎯 Ürün tavsiyeleri comma ile ayrıldı, dosyan hazır: Skin_text_classifier_urun_commalarli.csv")
'''
cilt_tipi_gundelik_mapping = {
    'hassas': 'Termal su spreyi, Bariyer onarıcı krem, Anti-Redness krem',
    'yagli': 'Salisilik asitli temizleyici, Oil-free nemlendirici, Matlaştırıcı tonik',
    'kuru': 'Seramid nemlendirici, Lipid bariyer krem, Yağ bazlı temizleyici'
}

df['cilt_onerilen_urun'] = df['cilttipi'].apply(lambda x: cilt_tipi_gundelik_mapping.get(x) if x in cilt_tipi_gundelik_mapping else None)

df.to_csv('Skin_text_classifier_son.csv', index=False, encoding='utf-8-sig')

print("🎯 Cilt tipine göre günlük ürün tavsiyesi eklendi (tum olanlar boş), dosyan hazır: Skin_text_classifier_full_plus.csv")