const { app, BrowserWindow, Menu, shell } = require("electron");
const path = require("path");

// ── Configuration ──────────────────────────────────────────
const FRONTEND_URL = process.env.RETINAQ_URL || "http://localhost:3000";
const DEV_MODE = process.argv.includes("--dev");

let mainWindow = null;

// ── Window Creation ────────────────────────────────────────
function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1440,
        height: 900,
        minWidth: 1024,
        minHeight: 700,
        title: "RETINA-Q — Quantum Retinal Diagnostic System",
        icon: path.join(__dirname, "icon.png"),
        backgroundColor: "#ffffff",
        webPreferences: {
            preload: path.join(__dirname, "preload.js"),
            nodeIntegration: false,
            contextIsolation: true,
            sandbox: false,
        },
    });

    // Retry loading until frontend is available
    const loadWithRetry = (retries = 30) => {
        mainWindow.loadURL(FRONTEND_URL).catch(() => {
            if (retries > 0) {
                setTimeout(() => loadWithRetry(retries - 1), 2000);
            } else {
                mainWindow.loadURL(
                    `data:text/html,<html><body style="background:#111;color:#0f0;font-family:monospace;display:flex;align-items:center;justify-content:center;height:100vh;margin:0"><div style="text-align:center"><h1>RETINA-Q</h1><p>Waiting for services...</p><p>Start the stack: <code>sudo docker compose up -d</code></p><p>Then reload: Ctrl+R</p></div></body></html>`
                );
            }
        });
    };
    loadWithRetry();

    if (DEV_MODE) {
        mainWindow.webContents.openDevTools({ mode: "detach" });
    }

    mainWindow.on("closed", () => {
        mainWindow = null;
    });

    // Open external links in the system browser
    mainWindow.webContents.setWindowOpenHandler(({ url }) => {
        shell.openExternal(url);
        return { action: "deny" };
    });
}

// ── Application Menu ───────────────────────────────────────
function buildMenu() {
    const template = [
        {
            label: "File",
            submenu: [
                {
                    label: "New Diagnosis",
                    accelerator: "CmdOrCtrl+N",
                    click: () => mainWindow?.loadURL(FRONTEND_URL),
                },
                { type: "separator" },
                { role: "quit" },
            ],
        },
        {
            label: "View",
            submenu: [
                { role: "reload" },
                { role: "forceReload" },
                { type: "separator" },
                { role: "resetZoom" },
                { role: "zoomIn" },
                { role: "zoomOut" },
                { type: "separator" },
                { role: "togglefullscreen" },
                ...(DEV_MODE ? [{ type: "separator" }, { role: "toggleDevTools" }] : []),
            ],
        },
        {
            label: "Navigate",
            submenu: [
                {
                    label: "Diagnose",
                    accelerator: "CmdOrCtrl+1",
                    click: () => mainWindow?.loadURL(FRONTEND_URL),
                },
                {
                    label: "System Dashboard",
                    accelerator: "CmdOrCtrl+2",
                    click: () => mainWindow?.loadURL(`${FRONTEND_URL}/dashboard`),
                },
                {
                    label: "Documentation",
                    accelerator: "CmdOrCtrl+3",
                    click: () => mainWindow?.loadURL(`${FRONTEND_URL}/docs`),
                },
            ],
        },
        {
            label: "Help",
            submenu: [
                {
                    label: "About RETINA-Q",
                    click: () => {
                        const { dialog } = require("electron");
                        dialog.showMessageBox(mainWindow, {
                            type: "info",
                            title: "About RETINA-Q",
                            message: "RETINA-Q v2.0.0",
                            detail:
                                "Hybrid Quantum-Classical Neural Network\nfor Retinal Disease Diagnosis\n\nPennyLane + PyTorch + Next.js + Electron",
                        });
                    },
                },
            ],
        },
    ];

    Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

// ── App Lifecycle ──────────────────────────────────────────
app.whenReady().then(() => {
    buildMenu();
    createWindow();

    app.on("activate", () => {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

app.on("window-all-closed", () => {
    if (process.platform !== "darwin") app.quit();
});
