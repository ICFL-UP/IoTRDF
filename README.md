# IoTRDF

The IoT Ransomware Detection Framework (IRDF) is a reinforcement learning–based system for selecting ultra-lean feature subsets to detect ransomware in IoT environments. It uses a Soft Actor–Critic (SAC) agent, a Python backend, a React-based frontend, and MySQL integration. An optional Docker-based virtual IoT testbed is included for latency and throughput experiments.

## Features
- SAC agent for automated feature selection
- Ultra-lean feature subsets 
- Python backend for training, evaluation, and policy management
- React-based web interface with real-time diagnostics
- MySQL database for experiment storage
- Optional Docker/VM testbed for inference performance evaluation

## System Components
### Backend (Python)
- SAC training engine
- Reward, state, and action definitions
- Classifier evaluation (DT, RF, LR, SVM, KNN)
- REST API for training and testing

### Frontend (React)
- Start/stop training
- Live logs: Q-values, feature probabilities, entropy, selected subset
- Policy display and testing

### Virtual IoT Testbed (Optional)
- Linux VM + Docker container for latency and throughput measurement

## Installation
IRDF runs primarily on a Windows 11 host system. The backend, frontend, and database run on Windows; the optional Docker-based testbed runs in a Linux virtual machine.
### 1. Clone the repository
git clone https://github.com/ICFL-UP/IoTRDF.git
cd IoTRDF
