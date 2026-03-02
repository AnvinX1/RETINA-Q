const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
    isElectron: true,
    platform: process.platform,
    appVersion: "2.0.0",
});
