// Canvas setup
const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
const predictionValue = document.getElementById('prediction-value');
const brushBtn = document.getElementById('brush');
const eraserBtn = document.getElementById('eraser');
const brushSizeSlider = document.getElementById('brush-size');
const sizeValue = document.getElementById('size-value');
const clearBtn = document.getElementById('clear-canvas');

// Drawing state
let isDrawing = false;
let lastX = 0;
let lastY = 0;
let currentTool = 'brush';
let brushSize = 20;

// Set canvas context defaults
ctx.lineJoin = 'round';
ctx.lineCap = 'round';
ctx.strokeStyle = '#FFFFFF';
ctx.lineWidth = brushSize;

// Drawing functions
function startDrawing(e) {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    [lastX, lastY] = [
        e.clientX - rect.left,
        e.clientY - rect.top
    ];
}

function draw(e) {
    if (!isDrawing) return;

    const rect = canvas.getBoundingClientRect();
    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(currentX, currentY);
    ctx.stroke();

    [lastX, lastY] = [currentX, currentY];

    clearTimeout(window.predictionTimeout);
    window.predictionTimeout = setTimeout(makePrediction, 300);
}

function stopDrawing() {
    isDrawing = false;
    makePrediction();
}

// Tool selection
function selectTool(tool) {
    currentTool = tool;

    [brushBtn, eraserBtn].forEach(btn => btn.classList.remove('active'));

    if (tool === 'brush') {
        brushBtn.classList.add('active');
        ctx.strokeStyle = '#FFFFFF';
    } else {
        eraserBtn.classList.add('active');
        ctx.strokeStyle = '#000000';
    }
}

// Clear canvas
function clearCanvas() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    predictionValue.textContent = '?';
}

// Update brush size
function updateBrushSize(size) {
    brushSize = size;
    ctx.lineWidth = size;
    sizeValue.textContent = size;
}

// Initialize ONNX model
let model;
let modelReady = false;

async function initModel() {
    try {
        model = new onnx.InferenceSession();
        await model.loadModel('model.onnx');
        modelReady = true;
        console.log('Model loaded successfully');
    } catch (e) {
        console.error('Failed to load ONNX model:', e);
    }
}

// Preprocess canvas to match model input
function preprocessCanvas() {
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = 28;
    tempCanvas.height = 28;

    // Scale drawing into 28x28 and black background
    tempCtx.fillStyle = 'black';
    tempCtx.fillRect(0, 0, 28, 28);
    tempCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, 28, 28);

    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const { data } = imageData;

    const input = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
        // Normalize pixel: white stroke on black => foreground is high
        input[i] = data[i * 4] / 255.0;
    }

    return input;
}

// Run prediction
async function makePrediction() {
    if (!modelReady) {
        console.log('Model not ready');
        return;
    }

    try {
        const input = preprocessCanvas();
        const tensor = new onnx.Tensor(input, 'float32', [1, 1, 28, 28]);

        const outputMap = await model.run([tensor]);
        const output = outputMap.values().next().value.data;

        let maxProb = -Infinity;
        let predictedDigit = -1;

        const expSum = output.reduce((sum, val) => sum + Math.exp(val), 0);
        for (let i = 0; i < output.length; i++) {
            const probability = Math.exp(output[i]) / expSum;
            if (probability > maxProb) {
                maxProb = probability;
                predictedDigit = i;
            }
        }

        predictionValue.textContent = predictedDigit.toString();
    } catch (e) {
        console.error('Prediction error:', e);
    }
}

// Mouse and touch event listeners
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

canvas.addEventListener('touchstart', e => {
    e.preventDefault();
    const touch = e.touches[0];
    canvas.dispatchEvent(new MouseEvent('mousedown', {
        clientX: touch.clientX,
        clientY: touch.clientY
    }));
});

canvas.addEventListener('touchmove', e => {
    e.preventDefault();
    const touch = e.touches[0];
    canvas.dispatchEvent(new MouseEvent('mousemove', {
        clientX: touch.clientX,
        clientY: touch.clientY
    }));
});

canvas.addEventListener('touchend', e => {
    e.preventDefault();
    canvas.dispatchEvent(new MouseEvent('mouseup'));
});

// Tool selection
brushBtn.addEventListener('click', () => selectTool('brush'));
eraserBtn.addEventListener('click', () => selectTool('eraser'));

// Brush size
brushSizeSlider.addEventListener('input', () => {
    updateBrushSize(parseInt(brushSizeSlider.value));
});

// Clear canvas
clearBtn.addEventListener('click', clearCanvas);

// Init
window.onload = () => {
    clearCanvas();
    initModel();
};
