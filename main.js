
// Robust in-browser IPD with camera fixes + diagnostics
import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const { FaceLandmarker, FilesetResolver } = vision;


const FIXED_DISTANCE_CM = 30.0;
const DEFAULT_IRIS_CM = 1.17;
const LEFT_IRIS = [468, 469, 470, 471];
const RIGHT_IRIS = [473, 474, 475, 476];

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
function circleFrom2(a, b) { return { c: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], r: dist(a, b) / 2 }; }
function circleFrom3(a, b, c) {
    const A = b[0] - a[0], B = b[1] - a[1], C = c[0] - a[0], D = c[1] - a[1];
    const E = A * (a[0] + b[0]) + B * (a[1] + b[1]);
    const F = C * (a[0] + c[0]) + D * (a[1] + c[1]);
    const G = 2 * (A * (c[1] - b[1]) - B * (c[0] - b[0]));
    if (Math.abs(G) < 1e-6) return null;
    const cx = (D * E - B * F) / G, cy = (A * F - C * E) / G;
    const r = dist([cx, cy], a);
    return { c: [cx, cy], r };
}
function minEnclosingCircle(pts) {
    let best = { c: [0, 0], r: Infinity };
    const n = pts.length;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        const cand = circleFrom2(pts[i], pts[j]);
        if (pts.every(p => dist(p, cand.c) <= cand.r + 1e-3) && cand.r < best.r) best = cand;
    }
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) for (let k = j + 1; k < n; k++) {
        const cand = circleFrom3(pts[i], pts[j], pts[k]); if (!cand) continue;
        if (pts.every(p => dist(p, cand.c) <= cand.r + 1e-3) && cand.r < best.r) best = cand;
    }
    if (!isFinite(best.r)) {
        const cx = pts.reduce((s, p) => s + p[0], 0) / n, cy = pts.reduce((s, p) => s + p[1], 0) / n;
        const r = pts.reduce((s, p) => s + dist(p, [cx, cy]), 0) / n;
        best = { c: [cx, cy], r };
    }
    return best;
}

// UI refs
const wrap = document.getElementById('wrap');
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
const selCam = document.getElementById('camera');
const mirrorChk = document.getElementById('mirror');
const debugEl = document.getElementById('debug');

// State
let faceLandmarker = null;
let running = false;
let currentStream = null;
let f_px = safeNum(localStorage.getItem('ipd_fpx'));
let iris_cm = safeNum(localStorage.getItem('ipd_iris_cm'), DEFAULT_IRIS_CM);
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
function format(num, digits = 2) { return (num == null || !isFinite(num)) ? 'N/A' : num.toFixed(digits); }
function safeNum(v, fallback = null) { const n = parseFloat(v); return Number.isFinite(n) ? n : fallback; }
function log(msg) { debugEl.textContent = String(msg ?? ''); }

function drawHUD(text, points = []) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#00d1ff';
    points.forEach(p => { ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, Math.PI * 2); ctx.fill(); });
    hud.textContent = text;
}

function resizeCanvasToVideo() {
    const w = video.videoWidth || 1280;
    const h = video.videoHeight || 720;
    canvas.width = w;
    canvas.height = h;
}

// Camera handling
async function listCameras() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const cams = devices.filter(d => d.kind === 'videoinput');
        selCam.innerHTML = '';
        cams.forEach((d, i) => {
            const opt = document.createElement('option');
            opt.value = d.deviceId;
            opt.textContent = d.label || `Camera ${i + 1}`;
            selCam.appendChild(opt);
        });
        if (!cams.length) log('No cameras found.');
    } catch (e) { log('enumerateDevices failed: ' + e.message); }
}

function getConstraints() {
    const [w, h] = selRes.value.split('x').map(Number);
    const deviceId = selCam.value || undefined;
    // Prefer front camera on mobile unless a device is chosen
    const facingMode = deviceId ? undefined : 'user';
    return {
        audio: false,
        video: {
            ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode }),
            width: { ideal: w },
            height: { ideal: h },
            frameRate: { ideal: 30, max: 30 }
        }
    };
}

async function stopCurrentStream() {
    if (currentStream) {
        currentStream.getTracks().forEach(t => t.stop());
        currentStream = null;
    }
}

async function enableCamera() {
    try {
        await stopCurrentStream();
        // First, request any video to unlock device labels (iOS/Chrome privacy)
        if (!window._labelsUnlocked) {
            const tmp = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            tmp.getTracks().forEach(t => t.stop());
            window._labelsUnlocked = true;
            await listCameras();
        }
        // Then request with our constraints
        let stream;
        try {
            stream = await navigator.mediaDevices.getUserMedia(getConstraints());
        } catch (e) {
            // Fallback: try generic video if specific constraint fails
            log('Specific constraints failed, retrying generic. ' + e.message);
            stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        }
        currentStream = stream;
        video.srcObject = stream;

        // Wait for metadata before playing/sizing (fixes black screen)
        await new Promise(res => { video.onloadedmetadata = () => res(); });
        await video.play();

        resizeCanvasToVideo();
        drawHUD('Camera ready. Click "Calibrate f_px" (hold ~30cm).');
        btnCalibF.disabled = false;
        btnCalibIris.disabled = false;
        btnReset.disabled = false;
    } catch (err) {
        log('getUserMedia error: ' + err.message + '\nDid you allow camera access?');
        throw err;
    }
}

// MediaPipe
async function initFaceLandmarker() {
    try {
        const files = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        faceLandmarker = await FaceLandmarker.createFromOptions(files, {
            baseOptions: {
                modelAssetPath:
                    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numFaces: 1,
            minFaceDetectionConfidence: 0.5,
            minFacePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
            outputFaceBlendshapes: false
        });
    } catch (e) {
        log("FaceLandmarker init failed: " + e.message);
        throw e;
    }
}


function landmarksToPts(landmarks, idxs, w, h) {
    return idxs.map(i => [landmarks[i].x * w, landmarks[i].y * h]);
}
function irisCenterDiamPx(landmarks, idxs, w, h) {
    const pts = landmarksToPts(landmarks, idxs, w, h);
    const { c, r } = minEnclosingCircle(pts);
    return { cx: c[0], cy: c[1], d: 2 * r };
}

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

// Calibrations
async function calibrateFpx() {
    if (!faceLandmarker) return;
    const tEnd = performance.now() + 3000;
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
    const tEnd = performance.now() + 2000;
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
    drawHUD('Reset done. Recalibrate f_px.');
}

// Events
btnStart.onclick = async () => {
    try {
        btnStart.disabled = true;
        log('Starting…');
        await initFaceLandmarker();
        await enableCamera();
        running = true;
        requestAnimationFrame(mainLoop);
        log(''); // clear
    } catch (e) {
        btnStart.disabled = false;
    }
};
btnCalibF.onclick = calibrateFpx;
btnCalibIris.onclick = calibrateIris;
btnReset.onclick = resetAll;
fixedDistChk.onchange = e => useFixed = e.target.checked;

selRes.onchange = async () => {
    if (video.srcObject) {
        running = false;
        await enableCamera();
        running = true;
        requestAnimationFrame(mainLoop);
    }
};
selCam.onchange = async () => {
    if (video.srcObject) {
        running = false;
        await enableCamera();
        running = true;
        requestAnimationFrame(mainLoop);
    }
};
mirrorChk.onchange = () => {
    wrap.classList.toggle('mirror', mirrorChk.checked);
};

window.addEventListener('resize', resizeCanvasToVideo);

// Populate camera list once allowed
navigator.mediaDevices?.enumerateDevices && listCameras();
