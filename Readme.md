# IPD (Browser)

- On-device IPD estimation using MediaPipe Face Landmarker (web).
- f_px calibration at ~30 cm, optional personal iris size calibration.
- Median+MAD smoothing and warnings for off-axis gaze.

### Run locally
- Use any static server (camera needs https **or** `http://localhost`):
  - `npx serve` or VS Code “Live Server”.
- Open in Chrome/Safari, click **Enable Camera**.

### Deploy (GitHub Pages)
1. Create a **public** repo, e.g. `ipd-web`.
2. Add `index.html`, `main.js`, `README.md`.
3. Commit & push.
4. In **Settings → Pages**, set **Build from branch**, `main`/`root`.
5. Open the Pages URL on your phone. Grant camera permission.

### Notes
- Processing runs in-browser; no backend.
- Model & WASM from CDNs. You may host them yourself if you prefer.
