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
    
    // Make prediction after short delay to avoid too frequent predictions
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
    
    // Update UI
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
    ctx.clearRect(0, 0, canvas.width, canvas.height);
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
        console.error('Make sure your model.onnx file is in the same directory as your HTML file');
        console.error('Also check if you need CORS enabled to load local files');
    }
}

// Preprocessing the canvas image for model input
function preprocessCanvas() {
    // Resize the canvas to 28x28 for MNIST format
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    
    // Draw the original canvas scaled down to 28x28 and in grayscale
    tempCtx.fillStyle = 'black';
    tempCtx.fillRect(0, 0, 28, 28);
    tempCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, 28, 28);
    
    // Get image data (RGBA)
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const { data } = imageData;
    
    // Convert to grayscale and normalize (values between 0 and 1)
    const input = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
        // Only looking at the R channel (data[i*4]), as R=G=B in grayscale
        input[i] = data[i * 4] / 255.0;
    }
    
    return input;
}

// Make prediction using the model
async function makePrediction() {
    if (!modelReady) {
        console.log('Model not ready yet');
        return;
    }
    
    try {
        // Preprocess canvas data
        const input = preprocessCanvas();
        
        // Reshape to match model input dimensions [1, 1, 28, 28]
        const tensor = new onnx.Tensor(input, 'float32', [1, 1, 28, 28]);
        
        // Run inference
        const outputMap = await model.run([tensor]);
        const output = outputMap.values().next().value.data;
        
        // Find predicted class (digit)
        let maxProb = -Infinity;
        let predictedDigit = -1;
        
        // Apply softmax to get probabilities
        const expSum = output.reduce((sum, val) => sum + Math.exp(val), 0);
        
        for (let i = 0; i < output.length; i++) {
            const probability = Math.exp(output[i]) / expSum;
            
            if (probability > maxProb) {
                maxProb = probability;
                predictedDigit = i;
            }
        }
        
        // Update UI with prediction
        predictionValue.textContent = predictedDigit.toString();
        
    } catch (e) {
        console.error('Prediction error:', e);
    }
}

// Event listeners
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Touch events for mobile support
canvas.addEventListener('touchstart', e => {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent('mousedown', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
});

canvas.addEventListener('touchmove', e => {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent('mousemove', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
});

canvas.addEventListener('touchend', e => {
    e.preventDefault();
    const mouseEvent = new MouseEvent('mouseup');
    canvas.dispatchEvent(mouseEvent);
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

// Initialize
window.onload = () => {
    clearCanvas();
    initModel();
};