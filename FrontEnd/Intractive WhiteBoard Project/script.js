const canvas = document.getElementById('whiteboard');
const ctx = canvas.getContext('2d');
const stickyContainer = document.getElementById('stickyContainer');
const colorPicker = document.getElementById('colorPicker');
const brushSize = document.getElementById('brushSize');
const clearBtn = document.getElementById('clearBtn');
const downloadBtn = document.getElementById('downloadBtn');
const toolBtns = document.querySelectorAll('.tool-btn');

// Set canvas size
canvas.width = window.innerWidth;
canvas.height = window.innerHeight - 80;

let isDrawing = false;
let currentTool = 'pen';
let startX, startY;
let snapshot;

// Tool selection
toolBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        toolBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentTool = btn.dataset.tool;
        
        if (currentTool === 'sticky') {
            createStickyNote();
        }
    });
});

// Drawing functions
function startDraw(e) {
    if (currentTool === 'sticky') return;
    
    isDrawing = true;
    startX = e.offsetX;
    startY = e.offsetY;
    snapshot = ctx.getImageData(0, 0, canvas.width, canvas.height);
    
    if (currentTool === 'pen' || currentTool === 'eraser') {
        ctx.beginPath();
        ctx.moveTo(startX, startY);
    }
}

function draw(e) {
    if (!isDrawing) return;
    
    const currentX = e.offsetX;
    const currentY = e.offsetY;
    
    ctx.lineWidth = brushSize.value;
    ctx.lineCap = 'round';
    
    if (currentTool === 'pen') {
        ctx.strokeStyle = colorPicker.value;
        ctx.lineTo(currentX, currentY);
        ctx.stroke();
    } else if (currentTool === 'eraser') {
        ctx.strokeStyle = '#ffffff';
        ctx.lineTo(currentX, currentY);
        ctx.stroke();
    } else if (currentTool === 'rectangle') {
        ctx.putImageData(snapshot, 0, 0);
        ctx.strokeStyle = colorPicker.value;
        ctx.strokeRect(startX, startY, currentX - startX, currentY - startY);
    } else if (currentTool === 'circle') {
        ctx.putImageData(snapshot, 0, 0);
        ctx.strokeStyle = colorPicker.value;
        const radius = Math.sqrt(Math.pow(currentX - startX, 2) + Math.pow(currentY - startY, 2));
        ctx.beginPath();
        ctx.arc(startX, startY, radius, 0, 2 * Math.PI);
        ctx.stroke();
    }
}

function stopDraw() {
    isDrawing = false;
}

// Sticky note creation
function createStickyNote() {
    const sticky = document.createElement('div');
    sticky.className = 'sticky-note';
    sticky.style.left = Math.random() * (window.innerWidth - 250) + 'px';
    sticky.style.top = Math.random() * (window.innerHeight - 250) + 100 + 'px';
    
    const textarea = document.createElement('textarea');
    textarea.placeholder = 'Type your note...';
    
    const deleteBtn = document.createElement('button');
    deleteBtn.className = 'sticky-delete';
    deleteBtn.textContent = '×';
    deleteBtn.onclick = () => sticky.remove();
    
    sticky.appendChild(deleteBtn);
    sticky.appendChild(textarea);
    stickyContainer.appendChild(sticky);
    
    makeDraggable(sticky);
}

// Make sticky notes draggable
function makeDraggable(element) {
    let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
    
    element.onmousedown = dragMouseDown;
    
    function dragMouseDown(e) {
        if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'BUTTON') return;
        e.preventDefault();
        pos3 = e.clientX;
        pos4 = e.clientY;
        document.onmouseup = closeDragElement;
        document.onmousemove = elementDrag;
    }
    
    function elementDrag(e) {
        e.preventDefault();
        pos1 = pos3 - e.clientX;
        pos2 = pos4 - e.clientY;
        pos3 = e.clientX;
        pos4 = e.clientY;
        element.style.top = (element.offsetTop - pos2) + 'px';
        element.style.left = (element.offsetLeft - pos1) + 'px';
    }
    
    function closeDragElement() {
        document.onmouseup = null;
        document.onmousemove = null;
    }
}

// Clear canvas
clearBtn.addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    stickyContainer.innerHTML = '';
});

// Download canvas
downloadBtn.addEventListener('click', () => {
    const link = document.createElement('a');
    link.download = 'whiteboard.png';
    link.href = canvas.toDataURL();
    link.click();
});

// Event listeners
canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDraw);
canvas.addEventListener('mouseout', stopDraw);

// Resize canvas
window.addEventListener('resize', () => {
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight - 80;
    ctx.putImageData(imgData, 0, 0);
});