// IPD web port of your Python script
// Keys: Buttons instead of keyboard. All on-device. No uploads.

const FIXED_DISTANCE_CM = 30.0;
const DEFAULT_IRIS_CM = 1.17;

// Iris landmark indices (refined mesh; same as your Python)
const LEFT_IRIS = [468, 469, 470, 471];
const RIGHT_IRIS = [473, 474, 475, 476];

// Smoother (median with MAD-based outlier rejection)
class RobustStream {
    constructor(win = 21, k = 3.5) { this.win = win; this.k = k; this.buf = []; }
    add(x) {
        if (x == null) return this.last();
        if (this.buf.length >= 5) {
            const med = median(this.buf);
            const mad = 1.4826 * median(this.buf.map(v => Math.abs(v - med)));
            const thresh = this.k * (mad > 1e-6 ? mad : 1.0);
            if (Math.abs(x - med) > thresh) return this.last();
        }
        if (this.buf.length >= this.win) this.buf.shift();
        this.buf.push(x);
        return this.last();
    }
    last() { return this.buf.length ? median(this.buf) : null; }
    clear() { this.buf = []; }
}
function median(arr) { const a = [...arr].sort((x, y) => x - y); const m = a.length >> 1; return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2; }
function dist(a, b) { const dx = a[0] - b[0], dy = a[1] - b[1]; return Math.hypot(dx, dy); }

// Minimal enclosing circle for 4 points (pairs + triplets); good enough for iris ring
function circleFrom2(a, b) { return { c: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], r: dist(a, b) / 2 }; }
function circleFrom3(a, b, c) {
    const A = b[0] - a[0], B = b[1] - a[1], C = c[0] - a[0], D = c[1] - a[1];
    const E = A * (a[0] + b[0]) + B * (a[1] + b[1]);
    const F = C * (a[0] + c[0]) + D * (a[1] + c[1]);
    const G = 2 * (A * (c[1] - b[1]) - B * (c[0] - b[0]));
    if (Math.abs(G) < 1e-6) return null; // colinear
    const cx = (D * E - B * F) / G, cy = (A * F - C * E) / G;
    const r = dist([cx, cy], a);
    return { c: [cx, cy], r };
}
function minEnclosingCircle(pts) {
    let best = { c: [0, 0], r: Infinity };
    const n = pts.length;
    // Try circles from 2 points
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        const cand = circleFrom2(pts[i], pts[j]);
        if (pts.every(p => dist(p, cand.c) <= cand.r + 1e-3) && cand.r < best.r) best = cand;
    }
    // Try circles from 3 points
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) for (let k = j + 1; k < n; k++) {
        const cand = circleFrom3(pts[i], pts[j], pts[k]); if (!cand) continue;
        if (pts.every(p => dist(p, cand.c) <= cand.r + 1e-3) && cand.r < best.r) best = cand;
    }
    // Fallback: centroid avg radius
    if (!isFinite(best.r)) {
        const cx = pts.reduce((s, p) => s + p[0], 0) / n, cy = pts.reduce((s, p) => s + p[1], 0) / n;
        const r = pts.reduce((s, p) => s + dist(p, [cx, cy]), 0) / n;
        best = { c: [cx, cy], r };
    }
    return best;
}

// UI
const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const hud = document.getElementById('hud');
const ctx = canvas.getContext('2d');

const btnStart = document.getElementById('btnStart');
const btnCalibF = document.getElementById('btnCalibF');
const btnCalibIris = document.getElementById('btnCalibIris');
const btnReset = document.getElementById('btnReset');
const fixedDistChk = document.getElementById('fixedDist');
const selRes = document.getElementById('res');

// State
let faceLandmarker = null;
let running = false;
let f_px = parseFloat(localStorage.getItem('ipd_fpx') || 'NaN'); if (!isFinite(f_px)) f_px = null;
let iris_cm = parseFloat(localStorage.getItem('ipd_iris_cm') || `${DEFAULT_IRIS_CM}`);
let useFixed = false;

const streamIris = new RobustStream(21, 3.5);
const streamIPD = new RobustStream(21, 3.5);

let procT = [];
function tick() {
    const now = performance.now();
    procT.push(now);
    if (procT.length > 60) procT.shift();
}
function procFps() {
    if (procT.length < 2) return 0;
    const dt = (procT[procT.length - 1] - procT[0]) / 1000;
    return (procT.length - 1) / dt;
}

// Setup MediaPipe
async function initFaceLandmarker() {
    const { FaceLandmarker, FilesetResolver } = window;
    const files = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(files, {
        baseOptions: {
            // Public model asset (float16). You can host locally if you prefer.
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
            delegate: 'GPU'
        },
        runningMode: 'VIDEO',
        numFaces: 1,
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
        outputFaceBlendshapes: false
    });
}

function getConstraints() {
    const [w, h] = selRes.value.split('x').map(Number);
    return {
        audio: false,
        video: {
            facingMode: 'user',
            width: { ideal: w },
            height: { ideal: h },
            frameRate: { ideal: 30, max: 30 }
        }
    };
}

async function enableCamera() {
    const stream = await navigator.mediaDevices.getUserMedia(getConstraints());
    video.srcObject = stream;
    await video.play();
    resizeToVideo();
}

function resizeToVideo() {
    const w = video.videoWidth || 1280;
    const h = video.videoHeight || 720;
    // letterbox to fit viewport while preserving aspect
    const vw = Math.min(window.innerWidth, w);
    const vh = Math.min(window.innerHeight - 150, h);
    video.width = canvas.width = w;
    video.height = canvas.height = h;
    // scale via CSS by aspect center—already handled by absolute centering
}

function landmarksToPts(landmarks, idxs, w, h) {
    return idxs.map(i => [landmarks[i].x * w, landmarks[i].y * h]);
}

function irisCenterDiamPx(landmarks, idxs, w, h) {
    const pts = landmarksToPts(landmarks, idxs, w, h);
    const { c, r } = minEnclosingCircle(pts);
    return { cx: c[0], cy: c[1], d: 2 * r };
}

function drawHUD(text, points = []) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // points (iris centers)
    ctx.fillStyle = '#00d1ff';
    points.forEach(p => {
        ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, Math.PI * 2); ctx.fill();
    });
    hud.textContent = text;
}

function format(num, digits = 2) { return (num == null || !isFinite(num)) ? 'N/A' : num.toFixed(digits); }

async function mainLoop() {
    if (!running || !faceLandmarker) return;
    tick();

    const w = video.videoWidth, h = video.videoHeight;
    const res = faceLandmarker.detectForVideo(video, performance.now());
    let irisPxInst = null, ipdPxInst = null, warn = '';

    if (res?.faceLandmarks?.length) {
        const lm = res.faceLandmarks[0];
        const L = irisCenterDiamPx(lm, LEFT_IRIS, w, h);
        const R = irisCenterDiamPx(lm, RIGHT_IRIS, w, h);

        if (L.d > 0 && R.d > 0) {
            const ratio = Math.max(L.d, R.d) / Math.max(1e-6, Math.min(L.d, R.d));
            if (ratio > 1.15) warn = 'off-axis gaze (iris mismatch)';
            else irisPxInst = 0.5 * (L.d + R.d);
        }
        ipdPxInst = Math.hypot(L.cx - R.cx, L.cy - R.cy);

        const smIris = streamIris.add(irisPxInst);
        const smIPD = streamIPD.add(ipdPxInst);

        const distance_cm = useFixed
            ? FIXED_DISTANCE_CM
            : (f_px && smIris) ? (f_px * iris_cm) / smIris : null;

        const ipd_cm = (f_px && distance_cm && smIPD)
            ? (smIPD * distance_cm) / f_px
            : null;

        drawHUD(
            [
                `Frame: ${w}x${h} | Proc FPS: ~${format(procFps(), 1)}`,
                f_px ? `f_px: ${format(f_px, 2)} px` : `f_px: N/A (calibrate)`,
                `iris_cm: ${format(iris_cm, 3)} cm${Math.abs(iris_cm - DEFAULT_IRIS_CM) > 1e-3 ? ' (personalized)' : ''}`,
                `Distance: ${distance_cm ? `${format(distance_cm, 2)} cm${useFixed ? ' (fixed)' : ' (est.)'}` : 'N/A'}`,
                `IPD: ${format(streamIPD.last(), 2)} px`,
                `IPD: ${format(ipd_cm, 2)} cm`,
                `Controls: Calibrate f_px • Calibrate iris • Reset • Fixed 30 cm`
            ].concat(warn ? [`Warn: ${warn}`] : []).join('\n'),
            [[L.cx, L.cy], [R.cx, R.cy]]
        );
    } else {
        drawHUD(`No face detected.\nProc FPS: ~${format(procFps(), 1)}\nControls: use the buttons below`);
    }

    requestAnimationFrame(mainLoop);
}

// Calibration routines
async function calibrateFpx() {
    if (!faceLandmarker) return;
    const tEnd = performance.now() + 3000; // 3s
    const samples = [];
    while (performance.now() < tEnd && samples.length < 20) {
        const w = video.videoWidth, h = video.videoHeight;
        const res = faceLandmarker.detectForVideo(video, performance.now());
        if (res?.faceLandmarks?.length) {
            const lm = res.faceLandmarks[0];
            const L = irisCenterDiamPx(lm, LEFT_IRIS, w, h);
            const R = irisCenterDiamPx(lm, RIGHT_IRIS, w, h);
            if (L.d > 0 && R.d > 0) {
                const ratio = Math.max(L.d, R.d) / Math.max(1e-6, Math.min(L.d, R.d));
                if (ratio <= 1.15) samples.push(0.5 * (L.d + R.d));
            }
        }
        await new Promise(r => setTimeout(r, 30));
        drawHUD(`Auto-calibrating f_px at ${FIXED_DISTANCE_CM.toFixed(1)} cm... ${samples.length}/20`);
    }
    if (samples.length >= 10) {
        const med = median(samples);
        f_px = (med * FIXED_DISTANCE_CM) / iris_cm;
        localStorage.setItem('ipd_fpx', String(f_px));
        streamIris.clear(); streamIPD.clear();
    } else {
        drawHUD(`Calibration failed. Try again with steady gaze at ~${FIXED_DISTANCE_CM} cm.`);
    }
}

async function calibrateIris() {
    if (!faceLandmarker || !f_px) { drawHUD('Calibrate f_px first.'); return; }
    const tEnd = performance.now() + 2000; // 2s
    const samples = [];
    while (performance.now() < tEnd && samples.length < 20) {
        const w = video.videoWidth, h = video.videoHeight;
        const res = faceLandmarker.detectForVideo(video, performance.now());
        if (res?.faceLandmarks?.length) {
            const lm = res.faceLandmarks[0];
            const L = irisCenterDiamPx(lm, LEFT_IRIS, w, h);
            const R = irisCenterDiamPx(lm, RIGHT_IRIS, w, h);
            if (L.d > 0 && R.d > 0) {
                const ratio = Math.max(L.d, R.d) / Math.max(1e-6, Math.min(L.d, R.d));
                if (ratio <= 1.15) samples.push(0.5 * (L.d + R.d));
            }
        }
        await new Promise(r => setTimeout(r, 30));
        drawHUD(`Calibrating personal iris size at fixed distance... ${samples.length}`);
    }
    if (samples.length >= 10) {
        const med = median(samples);
        iris_cm = (med * FIXED_DISTANCE_CM) / f_px;
        localStorage.setItem('ipd_iris_cm', String(iris_cm));
        streamIris.clear(); streamIPD.clear();
    } else {
        drawHUD('Iris calibration failed. Not enough good frames.');
    }
}

function resetAll() {
    f_px = null;
    iris_cm = DEFAULT_IRIS_CM;
    localStorage.removeItem('ipd_fpx');
    localStorage.setItem('ipd_iris_cm', String(iris_cm));
    streamIris.clear(); streamIPD.clear();
}

// Event handlers
btnStart.onclick = async () => {
    btnStart.disabled = true;
    await initFaceLandmarker();
    await enableCamera();
    running = true;
    requestAnimationFrame(mainLoop);
};
btnCalibF.onclick = calibrateFpx;
btnCalibIris.onclick = calibrateIris;
btnReset.onclick = resetAll;
fixedDistChk.onchange = e => useFixed = e.target.checked;
selRes.onchange = async () => {
    if (video.srcObject) {
        running = false;
        const tracks = video.srcObject.getTracks(); tracks.forEach(t => t.stop());
        await enableCamera();
        running = true;
        requestAnimationFrame(mainLoop);
    }
};
window.addEventListener('resize', resizeToVideo);
