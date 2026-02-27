# RETINA-Q Technology Stack

An exhaustive breakdown of the frameworks, libraries, and protocols powering the RETINA-Q system.

## 1. Machine Learning & Quantum Core
The entire AI backend operates on a Python 3.11+ stack.

- **PyTorch (v2.0+)**: The core deep learning framework utilized for defining the U-Net and EfficientNet-B0 architectures. Provides tensor manipulation and GPU-accelerated autograd.
- **PennyLane (v0.34+)**: An open-source cross-platform library for differentiable programming of quantum computers. We utilize PennyLane to build the 8-qubit and 4-qubit Variation Quantum Circuits (VQCs).
- **OpenCV-Python**: Handles critical spatial transformations, array manipulations, and color-space conversions (like CLAHE) required in medical imaging.
- **Torchvision**: Used for standard image transformations, normalization routines, and pre-trained backbone access.

## 2. Backend API
- **FastAPI**: A modern, fast (high-performance) web framework for building APIs. Chosen over Flask/Django for its async capabilities and automatic OpenAPI (Swagger) generation.
- **Pydantic**: Enforces strict typing and data validation for all incoming and outgoing JSON payloads.
- **Uvicorn**: An ASGI web server implementation used to run the FastAPI application concurrently in production.

## 3. Frontend UI/UX
- **Next.js (v14)**: The React framework for the web. Utilized for its robust App Router structure and performance optimizations.
- **Tailwind CSS**: A utility-first CSS framework for rapid UI styling.
- **Lucide React**: Provides sharp, minimalist SVG stroke icons.
- **Shadcn/UI Principles**: We hand-crafted components mimicking Shadcn's decoupled, completely un-styled primitive approach to ensure a stark black-and-white high-contrast vibe.
- **Capacitor (by Ionic)**: Enables the Next.js static HTML export to be securely packaged as a native iOS/Android mobile application frame.

## 4. Containerization & DevOps
- **Docker**: Isolates the node modules and python dependencies into stable, immutable images.
- **Docker Compose**: Orchestrates the multi-container stack, ensuring the frontend container can communicate internally via HTTP to the Python backend on Port 8000.
