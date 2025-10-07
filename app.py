from flask import Flask, render_template, request, jsonify
import face_recognition
import numpy as np
import base64
import cv2
import io
from PIL import Image

app = Flask(__name__)

# --- Simulasikan Database Pengguna (DATA PALSU) ---
# Di sistem nyata, ini diambil dari database. 
# Wajah yang sudah didaftarkan (face encoding) disimpan di sini.

# Misal: Wajah Budi memiliki ID unik (encoding) ini
# Anda harus mengganti array ini dengan face encoding wajah nyata
known_face_encodings = [
    np.array([0.1, 0.2, 0.3, ...]) # Ganti dengan encoding wajah Budi
]
known_face_names = [
    "Budi Santoso"
]
# ----------------------------------------------------

# Halaman utama (memuat index.html)
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint untuk memproses absensi
@app.route('/absensi', methods=['POST'])
def process_absensi():
    data = request.json
    image_data = data['image_data']

    try:
        # 1. Konversi data Base64 (dari web) kembali menjadi gambar
        # Hapus header 'data:image/jpeg;base64,'
        encoded_data = image_data.split(',')[1]
        decoded_data = base64.b64decode(encoded_data)
        
        # Buka gambar menggunakan PIL (Python Imaging Library)
        image = Image.open(io.BytesIO(decoded_data))
        # Konversi ke array numpy yang dapat diproses face_recognition
        face_frame = np.array(image)
        
        # 2. Deteksi dan Cari Encoding Wajah dalam gambar yang baru diterima
        face_locations = face_recognition.face_locations(face_frame)
        face_encodings = face_recognition.face_encodings(face_frame, face_locations)

        if not face_encodings:
            return jsonify({'status': 'failed', 'message': 'Gagal: Wajah tidak terdeteksi.'})

        # 3. Verifikasi Wajah (Face Matching)
        # Bandingkan wajah dari kamera dengan wajah di 'database' (known_face_encodings)
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Wajah Tidak Dikenali"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                
                # --- Logika Absensi Berhasil ---
                # Di sini Anda akan menyimpan catatan waktu absensi ke database sungguhan
                print(f"ABSEN BERHASIL untuk: {name} pada waktu sekarang.")
                return jsonify({'status': 'success', 'message': f'Absensi Berhasil! Selamat datang, {name}.'})

        # Jika loop selesai dan tidak ada kecocokan
        return jsonify({'status': 'failed', 'message': f'Gagal: {name}. Silakan coba lagi.'})

    except Exception as e:
        print(f"Error pada Back-End: {e}")
        return jsonify({'status': 'error', 'message': 'Terjadi kesalahan internal server.'})

if __name__ == '__main__':
    # Pastikan Anda sudah membuat file wajah 'known_face_encodings' 
    # sebelum menjalankan aplikasi ini di lingkungan produksi.
    app.run(debug=True)
