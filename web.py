from pywebio.input import actions, input
from pywebio.output import *
from pywebio import start_server
from illness_predictor import load_illness_model, predict_illness, load_cilt_model, predict_cilt
from problems_predictor import load_problem_model, predict_problem, load_ciltbert_model, predict_ciltbert
import pandas as pd

df_keras = pd.read_csv("Skin_text_classifier.csv")
df_bert = pd.read_csv("Skin.csv")

# 🌸 Öneri fonksiyonları
def öneri_keras(h, c):
    try:
        o1 = df_keras[df_keras["hastalik"] == h]["onerilen_urun"].values[0]
    except:
        o1 = "Yok"
    try:
        o2 = df_keras[df_keras["cilttipi"] == c]["cilt_onerilen_urun"].values[0]
    except:
        o2 = "Yok"
    return o1, o2

def öneri_bert(p, c):
    # Filtre: hem problem hem cilt tipi eşleşmeli
    filtre = df_bert[
        (df_bert['cilt_problemi'] == p) &
        (df_bert['cilttipi'] == c)
    ]
    if not filtre.empty:
        return filtre['onerilen_urun'].values[0]
    else:
        return "❌ Bu cilt tipi ve problem için öneri bulunamadı"

# ✨ Mesaj stili
def user_msg(msg):
    return f"""<div style='background:#ffe0f0;padding:10px;border-radius:10px;max-width:70%;margin-bottom:8px;box-shadow:1px 1px 5px #ccc;'>👤 <b>Sen:</b><br>{msg}</div>"""

def bot_msg(msg):
    return f"""<div style='background:#f2f9ff;padding:10px;border-radius:10px;max-width:70%;float:right;margin-bottom:8px;box-shadow:1px 1px 5px #ccc;'>🤖 <b>Asistan:</b><br>{msg}</div>"""

def urun_kartlari(urunler):
    return f"""<div style='margin-top:12px; background: #fff7f0; border:1px solid #ffd9c4; border-radius:10px; padding:12px; box-shadow:1px 1px 8px #eee;'>
    <b>🎁 Önerilen Ürünler:</b><br>{"<br>".join(f"• {u}" for u in urunler if u != "Yok")}
    </div>"""

# 💬 KERAS tabanlı chatbot
def keras_chat():
    clear()
    put_html("<h2 style='text-align:center;color:#8b5dff;'>🩺 Cilt Hastalığı Asistanı</h2>")
    put_markdown("_📌 Cilt hastalığınız olduğunu düşünüyorsanız, semptomlarınızı yazın._")

    model_h, tok_h, le_h = load_illness_model()
    model_c, tok_c, le_c = load_cilt_model()

    while True:
        msg = input("💬 Yaz:", placeholder="Boynumda yayılmaya başlayan kırmızı, kaşıntılı bir döküntü var gibi... ")
        if msg == "__bert__":
            return bert_chat()  # 💥 Geçiş
        with use_scope("chat", clear=False):
            put_html(user_msg(msg))

            hastalik, g_h = predict_illness(msg, model_h, tok_h, le_h)
            cilttipi, g_c = predict_cilt(msg, model_c, tok_c, le_c)
            o1, o2 = öneri_keras(hastalik, cilttipi)

            response = f"""
            🏥 <b>Hastalık:</b> {hastalik} ({g_h:.2f})<br>
            🧴 <b>Cilt Tipi:</b> {cilttipi} ({g_c:.2f})
            """
            put_html(bot_msg(response))
            put_html(urun_kartlari([o1, o2]))
            put_html("<div style='clear:both;'></div><hr>")

        # 🔁 Geçiş butonu
        if actions(label="Modeli değiştirmek ister misiniz?", buttons=["🌸 Genel Problemlere Geç", "Devam"]) == "🌸 Genel Problemlere Geç":
            return bert_chat()

# 💬 BERT tabanlı chatbot
def bert_chat():
    clear()
    put_html("<h2 style='text-align:center;color:#ff5da2;'>🌸 Genel Cilt Problemi Asistanı</h2>")
    put_markdown("_📌 Ciltle ilgili probleminizi yazın, öneriler sunalım._")

    model_p, tok_p, le_p = load_problem_model()
    model_c, tok_c, le_c = load_ciltbert_model()

    try:
        while True:
            msg = input("💬 Yaz:", placeholder="Yüzümde sivilceler çıktı, cildim yağlı gibi...")
            if msg == "__keras__":
                return keras_chat()  # 💥 Geçiş
            with use_scope("chat", clear=False):
                put_html(user_msg(msg))

                prob, g_p = predict_problem(msg, model_p, tok_p, le_p)
                cilttipi, g_c = predict_ciltbert(msg, model_c, tok_c, le_c)
                o1 = öneri_bert(prob, cilttipi)

                response = f"""
                🌡️ <b>Problemin:</b> {prob} ({g_p:.2f})<br>
                🧬 <b>Cilt Tipin:</b> {cilttipi} ({g_c:.2f})
                """
                put_html(bot_msg(response))
                put_html(urun_kartlari([o1]))
                put_html("<div style='clear:both;'></div><hr>")

            # 🔁 Geçiş butonu
            if actions(label="Modeli değiştirmek ister misiniz?", buttons=["🏥 Cilt Hastalığına Geç", "Devam"]) == "🏥 Cilt Hastalığına Geç":
                return keras_chat()

    except Exception as e:
        put_html(f"<div style='color:red;'>💥 Bir hata oluştu: {str(e)}</div>")


# 🧿 Ana seçim ekranı
def ana_menu():
    put_html("""
    <style>
        body {
            background: linear-gradient(135deg, #fff0f5, #ffffff);
            font-family: 'Segoe UI', sans-serif;
        }

        h1 {
            text-align: center;
            color: #ff69b4;
            font-size: 38px;
            margin-top: 40px;
            text-shadow: 1px 1px 2px #ccc;
        }

        .custom-button {
            background-color: #cceeff;
            color: #5e4b8b;
            font-size: 24px;
            font-weight: 600;
            padding: 50px 100px;
            border: none;
            border-radius: 9999px;
            margin: 20px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            cursor: pointer;
            transition: all 0.3s ease-in-out;
        }

        .custom-button:hover {
            background-color: #fddde9;
            transform: scale(1.05);
        }
    </style>
    """)

    put_html("<h1>💖 Cilt Sağlığı Asistanına Hoşgeldiniz 💖</h1>")
    put_html("<br><br>")
    put_text("🌼 Cilt sorununuza göre bir yol seçin:")
 
    def handle_choice(choice):
        if choice == "keras":
            keras_chat()
        elif choice == "bert":
            bert_chat()

    put_buttons([
        {"label": "🏥 Cilt hastalığınız olduğunu düşünüyorsanız", "value": "keras"},
        {"label": "🌸 Genel cilt problemleri için öneri almak istiyorsanız", "value": "bert"}
    ], onclick=handle_choice, link_style=False)

    put_html("</div>")
    put_html("<br><br>")

    put_html("<div class='footer'>💡 Bu araç bir tıbbi tanı sistemi değildir. Lütfen bir dermatoloğa danışın.</div>")



# 🚀 Başlat
if __name__ == "__main__":
    start_server(ana_menu, port=8080)
