# Turning RETINA-Q into a Mobile App (Capacitor)

Yes, RETINA-Q **will absolutely work** as a mobile app using Ionic Capacitor! Because we built the frontend entirely separated from the Python backend (no Next.js Server Components relying on Node API routes), the Next.js app can be cleanly exported to static HTML/JS/CSS and wrapped into a native iOS/Android application.

Here is the step-by-step guide to converting the current project:

## 1. Configure Next.js for Static Export
By default, Next.js runs as a Node server. Capacitor requires static assets.
Open `frontend/next.config.mjs` and change `output: 'standalone'` to `output: 'export'`:

```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
    output: 'export', // <-- CHANGE THIS
    images: {
        unoptimized: true, // Required for static export
    },
};

export default nextConfig;
```

## 2. Install Capacitor Dependencies
Navigate to the frontend folder and install Capacitor:
```bash
cd frontend
npm install @capacitor/core @capacitor/cli @capacitor/ios @capacitor/android
```

## 3. Initialize Capacitor
Initialize the Capacitor project inside your frontend directory:
```bash
npx cap init "Retina Q" "com.retinaq.app" --web-dir out
```
*Note: Make sure `--web-dir` is set to `out` (where Next.js drops the export files).*

## 4. Build and Sync
Every time you want to update the mobile app, you must build the Next.js project and sync it to Capacitor:
```bash
# Export the Next.js code to the 'out' folder
npm run build 

# Add your target platform (only needed once)
npx cap add android
npx cap add ios

# Sync the 'out' folder into the native mobile projects
npx cap sync
```

## 🚨 CRITICAL: The `localhost` Issue
Right now, `NEXT_PUBLIC_API_URL` is set to `http://localhost:8000`. 
If you run this on your phone, **your phone will try to talk to itself** (its own localhost), not your computer running the FastAPI backend!

Before running `npm run build` for mobile, you MUST change the API URL to your computer's local network IP address (e.g., `192.168.1.X`) or a public server URL.

**1. Create a `.env.production` file in `frontend/`:**
```env
# Replace with your computer's local IP or deployed domain
NEXT_PUBLIC_API_URL=http://192.168.1.100:8000
```

**2. Update Backend CORS:**
Make sure your Python backend allows the Capacitor origin.
In `backend/app/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "capacitor://localhost", # iOS
        "http://localhost",      # Android
        "*"                      # For testing
    ],
    # ...
)
```

## 5. Run on Device
To open the native IDEs to build to your physical phone:
```bash
npx cap open android  # Opens Android Studio
npx cap open ios      # Opens Xcode
```
