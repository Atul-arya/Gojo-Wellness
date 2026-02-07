# Gojo Wellness Tracker - Packaging Guide

## Quick Start (Browser)
```bash
# Run the dashboard
python run_dashboard.py

# Opens http://localhost:8080
```

## Desktop App (Electron)

### 1. Install Node.js and Electron
```bash
npm init -y
npm install electron --save-dev
```

### 2. Create main.js
```javascript
const { app, BrowserWindow } = require('electron')
const path = require('path')

function createWindow() {
    const win = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            nodeIntegration: true
        }
    })
    win.loadFile('dashboard/index.html')
}

app.whenReady().then(createWindow)
app.on('window-all-closed', () => app.quit())
```

### 3. Update package.json
```json
{
    "main": "main.js",
    "scripts": {
        "start": "electron ."
    }
}
```

### 4. Run
```bash
npm start
```

### 5. Build Executable
```bash
npm install electron-builder --save-dev
npx electron-builder
```

---

## Mobile (PWA - Works on Phone)

The dashboard already works as a PWA! Just:

1. Open `http://localhost:8080` on your phone (same WiFi)
2. Add to Home Screen
3. Works offline!

To find your PC's IP:
```bash
# Windows
ipconfig
# Look for IPv4 Address (e.g., 192.168.1.100)
```

Then open `http://192.168.1.100:8080` on phone.

---

## Android APK (Advanced)

For a native APK, use Capacitor:

```bash
npm install @capacitor/core @capacitor/cli
npx cap init Gojo com.gojo.wellness
npx cap add android
npx cap copy android
npx cap open android
```

Then build in Android Studio.

---

## Data Location

All your data stays local:
- `gojo_data/` - Your profiles and history
- `dashboard/` - The web interface

Nothing is sent to any server.
