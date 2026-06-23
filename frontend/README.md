# Argus AI - Frontend

This is the frontend Next.js application for **Argus AI** (Spatial Intelligence & Real-time Urban Anomaly Resolution Platform).

## Prerequisites

- **Node.js** (v18 or higher recommended)
- **npm** (Node Package Manager)

## How to Run Locally

To test the frontend, you must start the development server. Please ensure that the **backend server** is also running, as the frontend relies on its APIs.

### 1. Install Dependencies

In this `frontend` directory, install the required packages:

```bash
npm install
```

### 2. Start the Development Server

```bash
npm run dev
```

### 3. View the Application

Open your browser and navigate to [http://localhost:3000](http://localhost:3000) to access the interactive WebGL map and dashboard.

## Related Components
- **Backend**: You will need to start the FastAPI server located in the `../backend` directory for real-time anomaly detection and graph modeling features to work. Please refer to the root `README.md` for backend instructions.
