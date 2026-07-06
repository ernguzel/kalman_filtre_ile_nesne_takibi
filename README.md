# Kalman Filtresi ile Nesne Takibi

YOLO ile tespit edilen kişiyi (person) Kalman filtresi kullanarak video üzerinde takip eden basit bir proje.

## Nasıl Çalışır

1. **YOLO** her frame'de kişi tespiti yapar (`src/detector.py`).
2. **Kalman filtresi** tespit edilen bbox'ı (merkez, genişlik, yükseklik) kullanarak nesnenin konumunu tahmin eder ve yumuşatır (`src/kalman_bbox.py`).
3. **Tracker**, tespit gelmediği frame'lerde sadece Kalman tahminiyle takibe devam eder (`src/tracker.py`).
4. Sonuç video üzerine çizilip (ham tespit + Kalman kutusu + iz/trail) kaydedilir, ayrıca bir CSV log dosyası oluşturulur.

## Klasör Yapısı

- `configs/config.yaml` – tüm ayarlar (model yolu, video yolları, Kalman/tracker/çizim parametreleri)
- `src/` – detector, kalman filtresi, tracker, video işleme ve yardımcı fonksiyonlar
- `basic_examples/` – Kalman filtresinin basit 2D örnekleri (deneme amaçlı)
- `scripts/plot_kalman_log.py` – CSV log dosyalarından grafik çizen script
- `videos/input` – girdi videoları
- `videos/output` – çıktı videoları, log CSV'leri ve grafikler
- `models/` – YOLO model ağırlığı
- `main_kalman_test.py` – Kalman takibi ana çalıştırma dosyası
- `main_detection_test.py` – sadece YOLO tespiti test dosyası

## Kurulum

```bash
pip install ultralytics opencv-python pyyaml numpy matplotlib
```

## Kullanım

1. `configs/config.yaml` içinde `paths` altındaki model ve video yollarını kendine göre ayarla.
2. Kalman takibini çalıştır:

```bash
python main_kalman_test.py
```

3. Sadece YOLO tespitini görmek istersen:

```bash
python main_detection_test.py
```

4. Log CSV'sinden grafik çizmek için:

```bash
python scripts/plot_kalman_log.py
```

## Not

Şu an sadece tek kişi (single person) takibi destekleniyor.
