import electron from "electron";
import { GoogleGenAI } from "@google/genai";
import fs from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import os from "node:os";

const { app, BrowserWindow, ipcMain, session, nativeImage, Menu } = electron;

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");
const logPath = path.join(app.getPath("userData"), "hermes-seeker.log");

function logLine(...parts) {
  const line = `${new Date().toISOString()} ${parts.map((part) =>
    typeof part === "string" ? part : JSON.stringify(part)
  ).join(" ")}
`;
  try {
    fs.mkdirSync(path.dirname(logPath), { recursive: true });
    fs.appendFileSync(logPath, line);
  } catch {
    // Logging must never break voice startup.
  }
}

// Name the app "Hermes Seeker" (menu bar / about panel). The Dock tile fully reflects this
// only in a packaged build; in dev the generic Electron bundle name is used.
app.setName("Hermes Seeker");

const iconPath = path.join(repoRoot, "build", "icon.png");
const appIcon = fs.existsSync(iconPath) ? nativeImage.createFromPath(iconPath) : null;

function parseEnvFile(envPath) {
  if (!envPath || !fs.existsSync(envPath)) return;
  const contents = fs.readFileSync(envPath, "utf8");
  for (const rawLine of contents.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const equalsIndex = line.indexOf("=");
    if (equalsIndex === -1) continue;
    const key = line.slice(0, equalsIndex).trim();
    let value = line.slice(equalsIndex + 1).trim();
    if (!key || process.env[key]) continue;
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    process.env[key] = value;
  }
}

// Look for .env in several places so both the dev repo run and a packaged
// Hermes Seeker can find credentials. First match for a given key wins.
function loadEnvFile() {
  const candidates = [
    path.join(repoRoot, ".env"),
    path.join(os.homedir(), ".hermes-seeker", ".env"),
    process.resourcesPath ? path.join(process.resourcesPath, ".env") : null,
  ];
  for (const candidate of candidates) parseEnvFile(candidate);
}

loadEnvFile();

let mainWindow = null;
let liveSession = null;
let ai = null;
let liveStatus = { running: false, pid: null };
let userTranscriptBuffer = "";
let modelTranscriptBuffer = "";
const hermesRuns = new Map();
const pendingHermesAnnouncements = [];

function emitToRenderer(channel, payload) {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  mainWindow.webContents.send(channel, payload);
}

function emitEvent(event) {
  logLine("event", event.type, event);
  emitToRenderer("sidecar:event", { timestamp: Date.now() / 1000, ...event });
}

function flushTranscripts() {
  if (userTranscriptBuffer.trim()) {
    emitEvent({ type: "transcript", speaker: "you", text: userTranscriptBuffer.trim() });
  }
  if (modelTranscriptBuffer.trim()) {
    emitEvent({ type: "transcript", speaker: "gemini", text: modelTranscriptBuffer.trim() });
  }
  userTranscriptBuffer = "";
  modelTranscriptBuffer = "";
}

function hermesBaseUrl() {
  return process.env.HERMES_API_URL || "http://127.0.0.1:8642";
}

function hermesHeaders() {
  return {
    Authorization: `Bearer ${process.env.API_SERVER_KEY || "hermes-seeker-local-dev"}`,
    "Content-Type": "application/json",
  };
}

function userDisplayName() {
  return (process.env.SEEKER_USER_NAME || process.env.USER || process.env.USERNAME || "Alex").trim();
}

async function hermesRequest(method, pathName, body = undefined) {
  const response = await fetch(`${hermesBaseUrl()}${pathName}`, {
    method,
    headers: hermesHeaders(),
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await response.text();
  let json = {};
  if (text) {
    try {
      json = JSON.parse(text);
    } catch {
      json = { text };
    }
  }
  if (!response.ok) {
    throw new Error(`Hermes ${response.status}: ${text || response.statusText}`);
  }
  return json;
}

async function checkHermesStatus() {
  try {
    const health = await hermesRequest("GET", "/health");
    emitEvent({ type: "hermes_status", status: "ready", detail: health });
    return { reachable: true, health };
  } catch (error) {
    emitEvent({ type: "hermes_status", status: "error", error: error.message });
    return { reachable: false, error: error.message };
  }
}

async function submitHermesTask({ task, session_id = "hermes-seeker-voice", urgency = "normal" }) {
  if (!task || !String(task).trim()) {
    return { status: "error", error: "Task is required." };
  }
  const cleanTask = String(task).trim();
  emitEvent({ type: "hermes_task_update", status: "starting", task: cleanTask });
  const run = await hermesRequest("POST", "/v1/runs", {
    input: cleanTask,
    session_id,
    instructions:
      "You are invoked from Hermes Seeker voice. Work autonomously for safe, reversible tasks. If the user shares a vague link/file/repo/context, clarify the desired outcome before installing, changing, deleting, connecting accounts, spending money, or doing security-sensitive actions. Report concise final results.",
  });
  const runId = run.run_id || run.id;
  emitEvent({ type: "hermes_task_update", status: "started", task: cleanTask, run_id: runId, urgency });
  if (runId) watchHermesRun(runId, cleanTask);
  return { status: "started", run_id: runId, message: "Hermes has started the task." };
}

async function getHermesTaskStatus({ run_id }) {
  return hermesRequest("GET", `/v1/runs/${run_id}`);
}

async function stopHermesTask({ run_id }) {
  return hermesRequest("POST", `/v1/runs/${run_id}/stop`, {});
}

async function approveHermesAction({ run_id, choice }) {
  return hermesRequest("POST", `/v1/runs/${run_id}/approval`, { choice });
}

async function executeHermesTool(name, args = {}) {
  switch (name) {
    case "check_hermes_status":
      return checkHermesStatus();
    case "submit_hermes_task":
      return submitHermesTask(args);
    case "get_hermes_task_status":
      return getHermesTaskStatus(args);
    case "stop_hermes_task":
      return stopHermesTask(args);
    case "approve_hermes_action":
      return approveHermesAction(args);
    default:
      return { status: "error", error: `Unknown tool: ${name}` };
  }
}

async function watchHermesRun(runId, task) {
  if (hermesRuns.has(runId)) return;
  hermesRuns.set(runId, true);
  const terminal = new Set(["completed", "failed", "cancelled", "canceled", "error"]);
  let lastStatus = "";
  try {
    while (hermesRuns.has(runId)) {
      const run = await hermesRequest("GET", `/v1/runs/${runId}`);
      const status = String(run.status || "unknown");
      if (status !== lastStatus) {
        emitEvent({ type: "hermes_task_update", status, run_id: runId, task, run });
        lastStatus = status;
      }
      if (terminal.has(status)) {
        const output = run.output || run.final_response || "";
        emitEvent({ type: "hermes_task_update", status, run_id: runId, task, output });
        announceHermesCompletion({
          runId,
          task,
          status,
          output: String(output || "").slice(0, 2500),
        });
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
  } catch (error) {
    emitEvent({ type: "hermes_task_update", status: "error", run_id: runId, task, error: error.message });
  } finally {
    hermesRuns.delete(runId);
  }
}

function announceHermesCompletion({ runId, task, status, output }) {
  const eventText = [
    "SYSTEM_EVENT_HERMES_COMPLETE",
    `run_id: ${runId}`,
    `status: ${status}`,
    `original_task: ${task}`,
    "instructions_to_hermes_seeker:",
    `- Proactively tell ${userDisplayName()} Hermes has returned.`,
    "- If another conversation is in progress, politely pause it with a short bridge like: Quick update, Hermes is back with a result.",
    "- Give a concise spoken summary in 1-3 sentences.",
    "- Ask whether he wants to go through the details before continuing the current conversation.",
    "- Do not say you personally did the work; Hermes did.",
    "hermes_result:",
    output || "(Hermes returned no text output.)",
  ].join("\n");

  emitEvent({
    type: "hermes_completion",
    run_id: runId,
    task,
    status,
    output,
  });

  if (liveSession) {
    liveSession.sendRealtimeInput({ text: eventText });
  } else {
    pendingHermesAnnouncements.push(eventText);
  }
}

function buildHermesTools() {
  return [
    {
      functionDeclarations: [
        {
          name: "check_hermes_status",
          description: "Check if Hermes local API is reachable. Use this for questions about Hermes status.",
          parameters: { type: "object", properties: {} },
        },
        {
          name: "submit_hermes_task",
          description:
            "Immediately hand actionable work to Hermes. Invoke for deals, shopping, research, coding, file work, terminal tasks, summaries, automations, or anything requiring tools. Do not ask the user clarifying questions first. IMPORTANT: Hermes cannot see this voice conversation — the 'task' string is the ONLY context it gets. So you must write a complete, self-contained brief, not a short paraphrase.",
          parameters: {
            type: "object",
            properties: {
              task: {
                type: "string",
                description:
                  "A complete, self-contained task brief for Hermes written in clear English. Expand the user's spoken request into a precise instruction: include the goal, every concrete detail the user gave (names, numbers, URLs, dates, budgets, preferences, constraints), any sensible defaults you assumed, and the expected output/format. Do NOT compress it into a few words; write the full task as if Hermes has no prior context.",
              },
              session_id: { type: "string", description: "Stable session id. Default hermes-seeker-voice." },
              urgency: { type: "string", description: "low, normal, or high." },
            },
            required: ["task"],
          },
        },
        {
          name: "get_hermes_task_status",
          description: "Fetch the latest status for a Hermes run.",
          parameters: {
            type: "object",
            properties: { run_id: { type: "string" } },
            required: ["run_id"],
          },
        },
        {
          name: "stop_hermes_task",
          description: "Stop an active Hermes run.",
          parameters: {
            type: "object",
            properties: { run_id: { type: "string" } },
            required: ["run_id"],
          },
        },
        {
          name: "approve_hermes_action",
          description: "Resolve a Hermes approval request.",
          parameters: {
            type: "object",
            properties: {
              run_id: { type: "string" },
              choice: { type: "string", description: "once, session, always, or deny" },
            },
            required: ["run_id", "choice"],
          },
        },
      ],
    },
  ];
}

function buildLiveConfig() {
  return {
    responseModalities: ["AUDIO"],
    mediaResolution: "MEDIA_RESOLUTION_MEDIUM",
    speechConfig: {
      voiceConfig: {
        prebuiltVoiceConfig: {
          voiceName: process.env.GEMINI_LIVE_VOICE || "Zephyr",
        },
      },
    },
    contextWindowCompression: {
      triggerTokens: 104857,
      slidingWindow: { targetTokens: 52428 },
    },
    inputAudioTranscription: {},
    outputAudioTranscription: {},
    tools: buildHermesTools(),
    systemInstruction: {
      parts: [
        {
          text: [
            `You are Hermes Seeker, the realtime Seeker-native voice front-end for ${userDisplayName()}.`,
            "Hermes is your worker brain for tools, terminal, files, web, deals, coding, research, and automations.",
            "Do not use Gemini built-in Google Search in this build; it is disabled because quota-limited keys can close the Live session. Route current facts, web lookups, and real work to Hermes instead.",
            `Be decisive and useful. For safe, clear work, call submit_hermes_task immediately. For vague links/files/repos/context, or risky actions like installs, destructive edits, account connections, credentials, purchases, wallets, or production changes, ask one short clarification/confirmation first.`,
            "Routing rule: answer simple chat directly; route current facts, web lookups, multi-step work, monitoring, files, email, deals, coding, automation, or anything that should continue in the background -> Hermes.",
            `When you call submit_hermes_task, write the 'task' as a COMPLETE, self-contained brief. Hermes cannot hear this conversation, so do not send a short paraphrase. Expand what ${userDisplayName()} said into a precise, detailed instruction that captures the goal, every concrete detail mentioned (names, numbers, URLs, dates, budgets, preferences, constraints), any reasonable defaults you are assuming, and the expected result/format. Write it as if Hermes has zero prior context.`,
            `After submit_hermes_task returns, say one short acknowledgement like: On it, Hermes is handling that now. (Keep what you SAY to ${userDisplayName()} short, even though the task you SENT to Hermes is detailed.)`,
            `When you receive SYSTEM_EVENT_SESSION_START, immediately speak a warm welcome-back greeting to ${userDisplayName()} as instructed, without waiting for the user to talk first.`,
            `When you receive SYSTEM_EVENT_HERMES_COMPLETE, treat it as a high-priority background result from Hermes. Proactively announce it even if ${userDisplayName()} was chatting with you. Keep it polite and short: say Hermes is back, summarize the result, and ask whether they want to go through it before continuing.`,
            "Answer directly for greetings, quick chat, status questions, or when the safest next step is a brief clarification.",
            "Keep voice responses natural and short.",
          ].join("\n"),
        },
      ],
    },
  };
}

function sendWelcomeGreeting() {
  (async () => {
    let reachable = false;
    try {
      const status = await checkHermesStatus();
      reachable = Boolean(status.reachable);
    } catch {
      reachable = false;
    }
    if (!liveSession) return;

    const hermesLine = reachable
      ? "Hermes is online and all channels are connected, so we're good to go."
      : "I'm still bringing Hermes online, channels are connecting now.";

    const greeting =
      `SYSTEM_EVENT_SESSION_START: The session just started. Proactively greet ${userDisplayName()} out loud right now in a warm, concise way (1-2 sentences). ` +
      `Say something like: Hey ${userDisplayName()}, Hermes Seeker is awake. ${hermesLine} Then ask what they want to hunt down next. ` +
      "Speak this greeting immediately without waiting for the user to talk first.";

    liveSession.sendRealtimeInput({ text: greeting });
  })();
}

async function startLive() {
  logLine("startLive", "called");
  if (liveSession) return liveStatus;
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    emitEvent({ type: "fatal", message: "GEMINI_API_KEY is not set." });
    throw new Error("GEMINI_API_KEY is not set");
  }

  const model = process.env.GEMINI_LIVE_MODEL || "gemini-3.1-flash-live-preview";
  ai = new GoogleGenAI({ apiKey });
  emitEvent({ type: "sidecar_status", status: { running: true, model, mode: "webrtc-aec" } });
  emitEvent({ type: "gemini_status", status: "connecting", model });

  liveSession = await ai.live.connect({
    model,
    config: buildLiveConfig(),
    callbacks: {
      onopen() {
        logLine("gemini", "open", model);
        liveStatus = { running: true, pid: process.pid };
        emitEvent({ type: "sidecar_status", status: { running: true, pid: process.pid, model, mode: "webrtc-aec" } });
        emitEvent({ type: "gemini_status", status: "connected", model });
        emitEvent({ type: "audio_state", state: "listening" });
        while (pendingHermesAnnouncements.length > 0 && liveSession) {
          liveSession.sendRealtimeInput({ text: pendingHermesAnnouncements.shift() });
        }
        sendWelcomeGreeting();
      },
      onmessage(message) {
        logLine("gemini", "message", Object.keys(message || {}));
        handleLiveMessage(message);
      },
      onerror(error) {
        const detail = error?.message || String(error);
        logLine("gemini", "error", detail);
        emitEvent({ type: "fatal", message: "Gemini Live error", error: detail });
      },
      onclose(event) {
        const reason = event?.reason || "closed";
        logLine("gemini", "close", reason);
        flushTranscripts();
        liveSession = null;
        liveStatus = { running: false, pid: null };
        if (reason && reason !== "closed") {
          emitEvent({ type: "fatal", message: "Gemini Live closed", error: reason });
        } else {
          emitEvent({ type: "gemini_status", status: "offline" });
        }
        emitEvent({ type: "audio_state", state: "idle" });
        emitEvent({ type: "sidecar_status", status: liveStatus, reason });
      },
    },
  });

  return { running: true, pid: process.pid };
}

async function handleToolCall(toolCall) {
  const functionResponses = [];
  for (const call of toolCall.functionCalls || []) {
    emitEvent({ type: "tool_call", name: call.name, args: call.args || {} });
    try {
      const result = await executeHermesTool(call.name, call.args || {});
      functionResponses.push({ id: call.id, name: call.name, response: { result } });
    } catch (error) {
      functionResponses.push({
        id: call.id,
        name: call.name,
        response: { status: "error", error: error.message },
      });
    }
  }
  if (functionResponses.length && liveSession) {
    liveSession.sendToolResponse({ functionResponses });
  }
}

function handleLiveMessage(message) {
  if (message.toolCall) {
    handleToolCall(message.toolCall).catch((error) => {
      emitEvent({ type: "fatal", message: "Tool call failed", error: error.message });
    });
  }

  const content = message.serverContent;
  if (!content) return;

  if (content.interrupted) {
    flushTranscripts();
    emitToRenderer("live:interrupt", {});
    emitEvent({ type: "audio_state", state: "listening" });
    return;
  }

  if (content.inputTranscription?.text) userTranscriptBuffer += content.inputTranscription.text;
  if (content.outputTranscription?.text) modelTranscriptBuffer += content.outputTranscription.text;

  for (const part of content.modelTurn?.parts || []) {
    if (part.text) modelTranscriptBuffer += part.text;
    const inlineData = part.inlineData;
    if (!inlineData?.data) continue;
    const mimeType = inlineData.mimeType || "audio/pcm;rate=24000";
    if (!mimeType.startsWith("audio/")) continue;
    emitToRenderer("live:audio", { data: inlineData.data, mimeType });
    emitEvent({ type: "audio_state", state: "speaking" });
  }

  if (content.turnComplete) {
    flushTranscripts();
    emitEvent({ type: "audio_state", state: "listening" });
  }
}

async function stopLive() {
  if (liveSession) {
    try { liveSession.close(); } catch { /* ignore close races */ }
  }
  liveSession = null;
  liveStatus = { running: false, pid: null };
  emitToRenderer("live:interrupt", {});
  emitEvent({ type: "gemini_status", status: "offline" });
  emitEvent({ type: "audio_state", state: "idle" });
  emitEvent({ type: "sidecar_status", status: liveStatus });
  return liveStatus;
}

function sendAudioChunk(arrayBuffer) {
  if (!liveSession || !arrayBuffer) return;
  const buffer = Buffer.from(new Uint8Array(arrayBuffer));
  if (!buffer.byteLength) return;
  liveSession.sendRealtimeInput({
    audio: { data: buffer.toString("base64"), mimeType: "audio/pcm;rate=16000" },
  });
}

function sendCommand(command) {
  if (command?.type === "text" && command.text) {
    if (!liveSession) throw new Error("Gemini Live is not running");
    liveSession.sendRealtimeInput({ text: command.text });
  }
  if (command?.type === "submit_hermes_task" && command.task) {
    submitHermesTask({ task: command.task, session_id: command.session_id || "hermes-seeker-voice" }).catch((error) => {
      emitEvent({ type: "hermes_task_update", status: "error", task: command.task, error: error.message });
    });
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1180,
    height: 860,
    minWidth: 980,
    minHeight: 800,
    backgroundColor: "#050712",
    ...(appIcon ? { icon: appIcon } : {}),
    titleBarStyle: "hiddenInset",
    trafficLightPosition: { x: 14, y: 14 },
    vibrancy: "under-window",
    webPreferences: {
      preload: path.join(repoRoot, "electron", "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  const devUrl = process.env.VITE_DEV_SERVER_URL ?? "http://127.0.0.1:5173";
  const useProd = app.isPackaged || process.env.SEEKER_START_PROD === "1";
  if (useProd) mainWindow.loadFile(path.join(repoRoot, "dist", "index.html"));
  else mainWindow.loadURL(devUrl);
}

function installAppMenu() {
  if (process.platform !== "darwin") return;
  app.setAboutPanelOptions({
    applicationName: "Hermes Seeker",
    applicationVersion: app.getVersion(),
    ...(appIcon ? { iconPath } : {}),
  });
  const menu = Menu.buildFromTemplate([
    {
      label: "Hermes Seeker",
      submenu: [
        { role: "about" },
        { type: "separator" },
        { role: "hide" },
        { role: "hideOthers" },
        { role: "unhide" },
        { type: "separator" },
        { role: "quit" },
      ],
    },
    { role: "editMenu" },
    { role: "windowMenu" },
  ]);
  Menu.setApplicationMenu(menu);
}

app.whenReady().then(() => {
  if (appIcon && process.platform === "darwin" && app.dock) {
    app.dock.setIcon(appIcon);
  }
  installAppMenu();

  session.defaultSession.setPermissionRequestHandler((_wc, permission, callback) => {
    callback(permission === "media" || permission === "audioCapture" || permission === "videoCapture");
  });

  ipcMain.handle("sidecar:start", () => startLive());
  ipcMain.handle("sidecar:stop", () => stopLive());
  ipcMain.handle("sidecar:status", () => liveStatus);
  ipcMain.handle("sidecar:command", (_event, command) => sendCommand(command));
  ipcMain.on("live:audio", (_event, chunk) => sendAudioChunk(chunk));
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("before-quit", () => stopLive());
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
