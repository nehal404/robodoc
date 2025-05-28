const segmentModels = {
    ear: { model: 'models/ear_model/model.json', yaml: 'models/ear_model/metadata.yaml' },
    eye: { model: 'models/eye_model/model.json', yaml: 'models/eye_model/metadata.yaml' },
    skin: { model: 'models/skin_model/model.json', yaml: 'models/skin_model/metadata.yaml' },
    scalp: { model: 'models/scalp_model/model.json', yaml: 'models/scalp_model/metadata.yaml' },
    teeth: { model: 'models/oral_model/model.json', yaml: 'models/oral_model/metadata.yaml' }
};

function showBodyScan() { document.getElementById('bodyScan').style.display = 'block'; hideOtherSections(); }
function showRoboDocChat() { document.getElementById('roboDocChat').style.display = 'block'; hideOtherSections(); }
function showInvestCheckup() { document.getElementById('investCheckup').style.display = 'block'; hideOtherSections(); }
function hideOtherSections() {
    document.getElementById('bodyScan').style.display = 'none';
    document.getElementById('roboDocChat').style.display = 'none';
    document.getElementById('investCheckup').style.display = 'none';
}

async function loadModelAndLabels(modelPath, yamlPath) {
    const model = await tf.loadGraphModel(modelPath);
    const response = await fetch(yamlPath);
    const yamlText = await response.text();
    const labels = yamlText.split('\n').filter(line => line.trim()).map(line => line.trim());
    return { model, labels };
}

async function processImage() {
    const segment = document.getElementById('segmentSelect').value;
    const { model, labels } = await loadModelAndLabels(segmentModels[segment].model, segmentModels[segment].yaml);
    const file = document.getElementById('imageInput').files[0];
    if (!file) return;

    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = async () => {
        const canvas = document.getElementById('preview');
        canvas.style.display = 'block';
        canvas.width = 416;
        canvas.height = 416;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, 416, 416);

        let tensor = tf.browser.fromPixels(img)
            .resizeNearestNeighbor([416, 416])
            .toFloat()
            .div(tf.scalar(255.0))
            .expandDims();

        const output = model.predict(tensor);
        const probabilities = output.dataSync();
        const maxProb = Math.max(...probabilities);
        const classIndex = probabilities.indexOf(maxProb);
        const predictedClass = labels[classIndex] || 'Unknown';
        const confidence = (maxProb * 100).toFixed(2);

        document.getElementById('scanResult').textContent = `Prediction: ${predictedClass} (${confidence}%)`;
        URL.revokeObjectURL(img.src);
        tensor.dispose();
        output.dispose();
    };
}

document.getElementById('imageInput').addEventListener('change', processImage);

async function sendMessage() {
    const input = document.getElementById('chatInput').value;
    if (!input) return;

    const chatOutput = document.getElementById('chatOutput');
    chatOutput.innerHTML += `<p><strong>You:</strong> ${input}</p>`;

    try {
        const response = await fetch('https://your-proxy-url/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: input })
        });
        const data = await response.json();
        chatOutput.innerHTML += `<p><strong>RoboDoc:</strong> ${data.response}</p>`;
    } catch (error) {
        chatOutput.innerHTML += `<p><strong>RoboDoc:</strong> Error: Unable to process your request.</p>`;
    }
    document.getElementById('chatInput').value = '';
}

function processCheckup() {
    const name = document.getElementById('patientName').value;
    const age = document.getElementById('patientAge').value;
    if (name && age) {
        document.getElementById('checkupResult').textContent = `Checkup submitted for ${name}, Age: ${age}. Results pending.`;
    } else {
        document.getElementById('checkupResult').textContent = 'Please enter name and age.';
    }
}