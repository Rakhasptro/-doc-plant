// ========================================
// SISTEM PAKAR DIAGNOSA PENYAKIT TANAMAN
// Dengan Custom Trained Model
// ========================================

let model;
let classNames = [];
let currentImage;

// ========================================
// KNOWLEDGE BASE - Database Penyakit
// ========================================
const knowledgeBase = {
    // Contoh mapping - akan di-generate dari class_names.json
    // Format: 'class_name_from_model': { info penyakit }
};

// Mapping umum penyakit ke knowledge base detail
const diseaseInfo = {
    'early_blight': {
        severity: 'medium',
        description: 'Penyakit jamur Alternaria solani yang menyerang daun bagian bawah terlebih dahulu.',
        symptoms: [
            'Bercak coklat berbentuk lingkaran konsentris',
            'Daun menguning dan gugur',
            'Dimulai dari daun bagian bawah'
        ],
        treatment: [
            'Aplikasi fungisida mankozeb atau klorotalonil',
            'Buang dan musnahkan daun terinfeksi',
            'Perbaiki drainase dan sirkulasi udara',
            'Mulsa untuk mencegah percikan air dari tanah'
        ],
        prevention: [
            'Rotasi tanaman dengan tanaman non-solanaceae',
            'Jarak tanam yang cukup (60-90 cm)',
            'Hindari penyiraman dari atas',
            'Sanitasi kebun yang baik'
        ]
    },
    'late_blight': {
        severity: 'high',
        description: 'Penyakit serius Phytophthora infestans yang dapat menyebar sangat cepat.',
        symptoms: [
            'Bercak coklat kehijauan basah pada daun',
            'Lapisan putih seperti tepung di bawah daun',
            'Batang dan buah juga terinfeksi',
            'Penyebaran sangat cepat dalam kondisi lembab'
        ],
        treatment: [
            'SEGERA aplikasi fungisida sistemik (metalaksil, dimetomorf)',
            'Buang dan bakar tanaman terinfeksi parah',
            'Kurangi kelembaban dengan perbaiki drainase',
            'Aplikasi fungisida protektan secara preventif'
        ],
        prevention: [
            'Gunakan varietas tahan',
            'Monitoring rutin terutama saat musim hujan',
            'Aplikasi fungisida preventif',
            'Jarak tanam ideal untuk sirkulasi udara'
        ]
    },
    'leaf_spot': {
        severity: 'medium',
        description: 'Penyakit bercak daun yang disebabkan berbagai jenis jamur.',
        symptoms: [
            'Bercak coklat atau hitam pada daun',
            'Daun menguning di sekitar bercak',
            'Gugur daun prematur'
        ],
        treatment: [
            'Aplikasi fungisida berbasis tembaga atau mankozeb',
            'Buang daun yang terinfeksi',
            'Tingkatkan sirkulasi udara'
        ],
        prevention: [
            'Jarak tanam yang cukup',
            'Hindari penyiraman dari atas',
            'Sanitasi rutin'
        ]
    },
    'mosaic': {
        severity: 'high',
        description: 'Penyakit virus yang menyebabkan pola mosaik pada daun. Tidak ada obat kimia.',
        symptoms: [
            'Pola mosaik kuning-hijau pada daun',
            'Daun mengkerut dan berubah bentuk',
            'Pertumbuhan kerdil'
        ],
        treatment: [
            'TIDAK ADA OBAT - cabut dan musnahkan tanaman terinfeksi',
            'Kendalikan serangga vektor (kutu daun, thrips)',
            'Sterilisasi alat dengan alkohol 70%'
        ],
        prevention: [
            'Gunakan bibit bersertifikat bebas virus',
            'Kendalikan populasi serangga vektor',
            'Sterilisasi alat berkebun secara rutin'
        ]
    },
    'bacterial': {
        severity: 'high',
        description: 'Infeksi bakteri yang menyerang daun dan buah.',
        symptoms: [
            'Bercak hitam berminyak pada daun',
            'Halo kuning di sekitar bercak',
            'Gugur daun prematur'
        ],
        treatment: [
            'Bakterisida berbahan tembaga',
            'Buang bagian tanaman terinfeksi',
            'Hindari penyiraman overhead'
        ],
        prevention: [
            'Gunakan benih bersertifikat',
            'Rotasi tanaman',
            'Sanitasi alat berkebun'
        ]
    },
    'healthy': {
        severity: 'low',
        description: 'Tanaman dalam kondisi sehat, tidak menunjukkan gejala penyakit.',
        symptoms: [
            'Daun hijau segar tanpa bercak',
            'Pertumbuhan normal dan vigor',
            'Tidak ada perubahan warna abnormal'
        ],
        treatment: [
            'Pertahankan perawatan yang baik',
            'Lanjutkan monitoring rutin'
        ],
        prevention: [
            'Lanjutkan praktik budidaya yang baik',
            'Pemupukan berimbang',
            'Penyiraman teratur'
        ]
    }
};

// ========================================
// LOAD MODEL & CLASS NAMES
// ========================================
async function loadModel() {
    try {
        document.getElementById('modelStatus').innerHTML = '⏳ Memuat model AI...';
        document.getElementById('modelStatus').classList.add('loading');
        
        console.log('Loading custom trained model...');
        
        // Load class names first
        const classResponse = await fetch('class_names.json');
        classNames = await classResponse.json();
        console.log('Class names loaded:', classNames.length, 'classes');
        
        // Load TensorFlow.js model
        model = await tf.loadGraphModel('tfjs_model/model.json');
        console.log('Model loaded successfully!');
        
        // Warmup prediction
        const dummyInput = tf.zeros([1, 224, 224, 3]);
        model.predict(dummyInput).dispose();
        dummyInput.dispose();
        
        document.getElementById('modelStatus').innerHTML = `✅ Model AI siap! (${classNames.length} kelas penyakit)`;
        document.getElementById('modelStatus').classList.remove('loading');
        
    } catch (error) {
        console.error('Error loading model:', error);
        document.getElementById('modelStatus').innerHTML = '❌ Gagal memuat model: ' + error.message;
        alert('Gagal memuat model. Pastikan folder tfjs_model dan class_names.json ada di project Anda.');
    }
}

// ========================================
// DRAG & DROP FUNCTIONALITY
// ========================================
const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');

uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleImageUpload(file);
    } else {
        alert('File harus berupa gambar (JPG/PNG)');
    }
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleImageUpload(file);
    }
});

// ========================================
// HANDLE IMAGE UPLOAD
// ========================================
function handleImageUpload(file) {
    if (file.size > 5 * 1024 * 1024) {
        alert('Ukuran file terlalu besar! Maksimal 5MB');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        const img = document.getElementById('imagePreview');
        img.src = e.target.result;
        currentImage = img;
        
        document.getElementById('previewSection').classList.add('active');
        document.getElementById('resultSection').classList.remove('active');
    };
    reader.readAsDataURL(file);
}

// ========================================
// PREPROCESS IMAGE
// ========================================
function preprocessImage(img) {
    return tf.tidy(() => {
        // Convert image to tensor
        let tensor = tf.browser.fromPixels(img);
        
        // Resize to 224x224
        tensor = tf.image.resizeBilinear(tensor, [224, 224]);
        
        // Normalize to [0, 1]
        tensor = tensor.div(255.0);
        
        // Add batch dimension
        tensor = tensor.expandDims(0);
        
        return tensor;
    });
}

// ========================================
// ANALYZE IMAGE
// ========================================
async function analyzeImage() {
    if (!model) {
        alert('Model AI belum siap! Tunggu sebentar dan coba lagi.');
        return;
    }

    if (!currentImage) {
        alert('Belum ada gambar yang dipilih!');
        return;
    }

    document.getElementById('loading').classList.add('active');
    document.getElementById('analyzeBtn').disabled = true;

    try {
        // Preprocess image
        const tensor = preprocessImage(currentImage);
        
        // Make prediction
        const predictions = await model.predict(tensor).data();
        tensor.dispose();
        
        // Get top 3 predictions
        const top3 = getTop3Predictions(predictions);
        
        console.log('Top 3 predictions:', top3);
        
        // Display results
        displayResults(top3);
        
    } catch (error) {
        console.error('Error analyzing image:', error);
        alert('Terjadi kesalahan saat menganalisa gambar: ' + error.message);
    } finally {
        document.getElementById('loading').classList.remove('active');
        document.getElementById('analyzeBtn').disabled = false;
    }
}

// ========================================
// GET TOP 3 PREDICTIONS
// ========================================
function getTop3Predictions(predictions) {
    const predArray = Array.from(predictions);
    const indices = predArray
        .map((prob, index) => ({ index, prob }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, 3);
    
    return indices.map(item => ({
        className: classNames[item.index],
        probability: item.prob
    }));
}

// ========================================
// MAP CLASS TO DISEASE INFO
// ========================================
function mapClassToDisease(className) {
    const lowerClass = className.toLowerCase();
    
    // Cek keyword penyakit
    if (lowerClass.includes('early_blight') || lowerClass.includes('early blight')) {
        return { type: 'early_blight', name: extractPlantName(className) + ' - Early Blight' };
    }
    if (lowerClass.includes('late_blight') || lowerClass.includes('late blight')) {
        return { type: 'late_blight', name: extractPlantName(className) + ' - Late Blight' };
    }
    if (lowerClass.includes('leaf_spot') || lowerClass.includes('septoria') || lowerClass.includes('spot')) {
        return { type: 'leaf_spot', name: extractPlantName(className) + ' - Leaf Spot' };
    }
    if (lowerClass.includes('mosaic') || lowerClass.includes('virus')) {
        return { type: 'mosaic', name: extractPlantName(className) + ' - Mosaic Virus' };
    }
    if (lowerClass.includes('bacterial') || lowerClass.includes('bacteria')) {
        return { type: 'bacterial', name: extractPlantName(className) + ' - Bacterial Spot' };
    }
    if (lowerClass.includes('healthy')) {
        return { type: 'healthy', name: extractPlantName(className) + ' - Sehat' };
    }
    
    // Default
    return { type: 'leaf_spot', name: className };
}

// ========================================
// EXTRACT PLANT NAME
// ========================================
function extractPlantName(className) {
    // Extract plant name from class (e.g., "Tomato___Early_Blight" -> "Tomat")
    const plantMap = {
        'tomato': 'Tomat',
        'potato': 'Kentang',
        'pepper': 'Cabai',
        'corn': 'Jagung',
        'apple': 'Apel',
        'grape': 'Anggur',
        'strawberry': 'Stroberi',
        'peach': 'Persik',
        'cherry': 'Ceri'
    };
    
    const lowerClass = className.toLowerCase();
    for (const [eng, indo] of Object.entries(plantMap)) {
        if (lowerClass.includes(eng)) {
            return indo;
        }
    }
    
    return 'Tanaman';
}

// ========================================
// DISPLAY RESULTS
// ========================================
function displayResults(predictions) {
    const topPrediction = predictions[0];
    const confidence = (topPrediction.probability * 100).toFixed(1);
    
    // Map to disease info
    const diseaseMapping = mapClassToDisease(topPrediction.className);
    const diseaseType = diseaseMapping.type;
    const diseaseName = diseaseMapping.name;
    const diseaseData = diseaseInfo[diseaseType] || diseaseInfo['leaf_spot'];
    
    let html = `
        <div class="diagnosis-card ${diseaseData.severity === 'low' ? 'healthy' : ''}">
            <h3>${diseaseName}</h3>
            <div class="confidence">Tingkat Keyakinan: ${confidence}%</div>
            <div class="severity ${diseaseData.severity}">
                Keparahan: ${diseaseData.severity === 'high' ? 'Tinggi' : diseaseData.severity === 'medium' ? 'Sedang' : 'Rendah'}
            </div>

            <div class="info-section">
                <h4>📝 Deskripsi</h4>
                <p>${diseaseData.description}</p>

                <h4>🔍 Gejala yang Terdeteksi</h4>
                <ul>
                    ${diseaseData.symptoms.map(s => `<li>${s}</li>`).join('')}
                </ul>

                <h4>💊 Rekomendasi Penanganan</h4>
                <ul>
                    ${diseaseData.treatment.map(t => `<li>${t}</li>`).join('')}
                </ul>

                <h4>🛡️ Pencegahan</h4>
                <ul>
                    ${diseaseData.prevention.map(p => `<li>${p}</li>`).join('')}
                </ul>
            </div>
        </div>

        <div style="background: white; padding: 15px; border-radius: 10px; margin-top: 15px;">
            <h4 style="color: #34495e; margin-bottom: 10px;">🤖 Detail Deteksi AI (Top 3)</h4>
            <div style="font-size: 0.9em; color: #7f8c8d;">
                ${predictions.map((p, i) => {
                    const pct = (p.probability * 100).toFixed(1);
                    const barWidth = p.probability * 100;
                    return `
                        <div style="margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                                <span>${i + 1}. ${p.className}</span>
                                <strong>${pct}%</strong>
                            </div>
                            <div style="background: #ecf0f1; border-radius: 10px; height: 8px; overflow: hidden;">
                                <div style="background: linear-gradient(90deg, #3498db, #2ecc71); width: ${barWidth}%; height: 100%;"></div>
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
        </div>
    `;

    document.getElementById('diagnosisResult').innerHTML = html;
    document.getElementById('resultSection').classList.add('active');
}

// ========================================
// RESET APP
// ========================================
function resetApp() {
    document.getElementById('previewSection').classList.remove('active');
    document.getElementById('resultSection').classList.remove('active');
    document.getElementById('fileInput').value = '';
    currentImage = null;
}

// ========================================
// INITIALIZE
// ========================================
window.onload = loadModel;