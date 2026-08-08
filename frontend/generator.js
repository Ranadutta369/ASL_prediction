// ==========================================================================
// SIGN0 TEXT-TO-SIGN DIFFUSION ANIMATION RENDERER
// ==========================================================================

const promptInput = document.getElementById("promptInput");
const generateBtn = document.getElementById("generateBtn");
const animCanvas = document.getElementById("animCanvas");
const ctx = animCanvas.getContext("2d");

const playPauseBtn = document.getElementById("playPauseBtn");
const loopBtn = document.getElementById("loopBtn");
const speedSelect = document.getElementById("speedSelect");
const playerStatus = document.getElementById("playerStatus");

const LOCAL_HOSTS = ["127.0.0.1", "localhost"];
const LOCAL_PORTS = [8000, 8005, 8001, 8080];
const REMOTE_API_BASE = "https://asl-prediction-v2.onrender.com";
let activeApiBaseUrl = "http://127.0.0.1:8000";
let apiReady = false;

// MediaPipe Hand Connection Pairs
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // Index
  [5, 9], [9, 10], [10, 11], [11, 12],  // Middle
  [9, 13], [13, 14], [14, 15], [15, 16],// Ring
  [13, 17], [17, 18], [18, 19], [19, 20],// Pinky
  [0, 17]                               // Palm base
];

let currentAnimation = null; // Array of 3D frames (Time, 21, 3)
let currentFrameIndex = 0;
let isPlaying = false;
let isLooping = true;
let animTimer = null;
let playbackSpeed = 1.0;

// Auto-detect working local backend (tests 127.0.0.1 and localhost across ports)
async function checkBackendConnection() {
  for (const host of LOCAL_HOSTS) {
    for (const port of LOCAL_PORTS) {
      try {
        const url = `http://${host}:${port}`;
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 1000);
        const res = await fetch(`${url}/`, { method: "GET", signal: controller.signal });
        clearTimeout(timeoutId);
        if (res.ok) {
          activeApiBaseUrl = url;
          apiReady = true;
          return;
        }
      } catch (err) {
        // Continue checking
      }
    }
  }

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000);
    const res = await fetch(`${REMOTE_API_BASE}/`, { method: "GET", signal: controller.signal });
    clearTimeout(timeoutId);
    if (res.ok) {
      activeApiBaseUrl = REMOTE_API_BASE;
      apiReady = true;
      return;
    }
  } catch (err) {
    apiReady = false;
  }
}

function setPrompt(val) {
  promptInput.value = val;
  // Highlight active chip
  document.querySelectorAll("[data-prompt]").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.prompt.toLowerCase() === val.toLowerCase());
  });
  generateAnimation();
}

async function generateAnimation() {
  const promptText = promptInput.value.trim();
  if (!promptText) return;

  playerStatus.innerText = "Synthesizing keypoints...";
  playerStatus.style.color = "#38bdf8";

  try {
    if (!apiReady) {
      await checkBackendConnection();
    }

    if (!apiReady) {
      throw new Error("Backend API unavailable");
    }

    const res = await fetch(`${activeApiBaseUrl}/synthesize_sign`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: promptText }),
    });

    if (!res.ok) throw new Error(`API HTTP ${res.status}`);

    const data = await res.json();
    currentAnimation = data.keypoints_3d;
    currentFrameIndex = 0;

    const infoStr = data.words_synthesized ? data.words_synthesized.join(", ") : data.prompt;
    playerStatus.innerText = `Generated ${data.num_frames} frames [${infoStr}]`;
    playerStatus.style.color = "#22c55e";

    startPlayback();
  } catch (err) {
    console.warn("Backend API notice, generating local dynamic trajectory:", err);
    apiReady = false;
    currentAnimation = generateLocalTrajectory(promptText);
    currentFrameIndex = 0;
    playerStatus.innerText = `Local 3D trajectory (${promptText})`;
    playerStatus.style.color = "#a855f7";
    startPlayback();
  }
}

function startPlayback() {
  if (animTimer) clearInterval(animTimer);
  isPlaying = true;
  playPauseBtn.innerText = "Pause Animation";

  const frameIntervalMs = 70 / playbackSpeed;
  animTimer = setInterval(() => {
    if (!currentAnimation || currentAnimation.length === 0) return;

    renderFrame(currentAnimation[currentFrameIndex]);
    currentFrameIndex++;

    if (currentFrameIndex >= currentAnimation.length) {
      if (isLooping) {
        currentFrameIndex = 0;
      } else {
        stopPlayback();
      }
    }
  }, frameIntervalMs);
}

function stopPlayback() {
  isPlaying = false;
  if (animTimer) clearInterval(animTimer);
  playPauseBtn.innerText = "Play Animation";
}

function renderFrame(frame3D) {
  const W = animCanvas.width;
  const H = animCanvas.height;
  const cx = W / 2;
  const cy = H / 2 + 20;

  ctx.clearRect(0, 0, W, H);

  // Background Grid Glow
  ctx.strokeStyle = "rgba(56, 189, 248, 0.06)";
  ctx.lineWidth = 1;
  for (let x = 0; x < W; x += 40) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }
  for (let y = 0; y < H; y += 40) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }

  if (!frame3D || frame3D.length < 21) return;

  // Auto-center landmark coordinates around wrist (joint 0)
  const wristX = frame3D[0][0];
  const wristY = frame3D[0][1];
  const wristZ = frame3D[0][2];

  const centered = frame3D.map(([x, y, z]) => [x - wristX, y - wristY, z - wristZ]);

  // Compute maximum distance from wrist for dynamic scaling
  let maxDist = 0;
  for (const [x, y, z] of centered) {
    const dist = Math.sqrt(x * x + y * y + z * z);
    if (dist > maxDist) maxDist = dist;
  }
  if (maxDist < 1e-6) maxDist = 1.0;

  const scale = 170 / maxDist; // Fit comfortably inside canvas

  // Project 3D to 2D centered on canvas
  const coords2D = centered.map(([x, y, z]) => [cx + x * scale, cy + y * scale]);

  // Draw 3D Connectors
  ctx.lineWidth = 4.0;
  ctx.strokeStyle = "#38bdf8"; // Neon Cyan
  ctx.shadowColor = "#38bdf8";
  ctx.shadowBlur = 12;

  for (const [i, j] of HAND_CONNECTIONS) {
    const [x1, y1] = coords2D[i];
    const [x2, y2] = coords2D[j];
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }

  // Draw Joint Spheres
  ctx.shadowBlur = 0;
  for (let i = 0; i < coords2D.length; i++) {
    const [x, y] = coords2D[i];
    ctx.beginPath();
    ctx.arc(x, y, i === 0 ? 8.5 : 5, 0, 2 * Math.PI);
    ctx.fillStyle = i === 0 ? "#ef4444" : "#22c55e"; // Red wrist, green joints
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // Frame Counter Badge
  ctx.fillStyle = "rgba(255, 255, 255, 0.5)";
  ctx.font = "12px Inter, sans-serif";
  ctx.fillText(`Frame ${currentFrameIndex + 1} / ${currentAnimation.length}`, 16, 28);
}

// Fallback dynamic trajectory generator for any text
function generateLocalTrajectory(promptText) {
  const words = promptText.toLowerCase().trim().split(/\s+/);
  const frames = [];

  const basePalm = [
    [0.0, 0.0, 0.0], [0.08, -0.05, 0.0], [0.14, -0.12, 0.0], [0.18, -0.18, 0.0], [0.22, -0.22, 0.0],
    [0.05, -0.25, 0.0], [0.07, -0.35, 0.0], [0.08, -0.42, 0.0], [0.09, -0.48, 0.0],
    [0.0, -0.26, 0.0], [0.0, -0.37, 0.0], [0.0, -0.45, 0.0], [0.0, -0.52, 0.0],
    [-0.05, -0.24, 0.0], [-0.07, -0.34, 0.0], [-0.08, -0.41, 0.0], [-0.09, -0.47, 0.0],
    [-0.10, -0.21, 0.0], [-0.13, -0.29, 0.0], [-0.15, -0.35, 0.0], [-0.17, -0.40, 0.0]
  ];

  for (const word of words) {
    const hashVal = word.split("").reduce((acc, char) => acc + char.charCodeAt(0), 0);
    const waveFreq = 1.0 + (hashVal % 3) * 0.5;

    for (let t = 0; t < 20; t++) {
      const progress = t / 20.0;
      const wave = Math.sin(progress * Math.PI * 2 * waveFreq) * 0.07;
      const lift = Math.sin(progress * Math.PI) * 0.09;

      const frame = basePalm.map(([x, y, z], idx) => {
        const dx = (idx >= 5) ? wave : wave * 0.5;
        const dy = (idx >= 5) ? -lift : 0;
        return [x + dx, y + dy, z];
      });
      frames.push(frame);
    }
  }

  return frames;
}

// Event Listeners
generateBtn.addEventListener("click", generateAnimation);

promptInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    generateAnimation();
  }
});

playPauseBtn.addEventListener("click", () => {
  if (isPlaying) {
    stopPlayback();
  } else {
    startPlayback();
  }
});

loopBtn.addEventListener("click", () => {
  isLooping = !isLooping;
  loopBtn.classList.toggle("active", isLooping);
  loopBtn.innerText = isLooping ? "Loop: ON" : "Loop: OFF";
});

speedSelect.addEventListener("change", (e) => {
  playbackSpeed = parseFloat(e.target.value);
  if (isPlaying) startPlayback();
});

document.querySelectorAll("[data-prompt]").forEach((btn) => {
  btn.addEventListener("click", () => setPrompt(btn.dataset.prompt));
});

// Auto generate on page load
checkBackendConnection().finally(generateAnimation);

