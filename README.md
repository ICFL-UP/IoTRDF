# IoTRDF

The IoT Ransomware Detection Framework (IRDF) is a reinforcement learning–driven system designed to detect ransomware in IoT environments using an optimised subset of network-traffic features. The framework employs a hierarchical Soft Actor–Critic (SAC) agent to perform automatic feature selection, significantly reducing dimensionality while preserving high detection accuracy. IRDF includes a web-based interface for policy training and evaluation, along with an optional Docker-based virtual IoT testbed for latency and throughput measurements.

---

##  Features

- Reinforcement learning–based feature selection using hierarchical SAC  
- Selection of ultra-lean feature subsets (for example, 1–3 features)  
- Python backend for model training, evaluation, and policy storage  
- Web-based user interface for managing experiments and visualising logs  
- MySQL database integration for results and configuration  
- Optional Linux-based Docker testbed for real-time inference experiments  
- Modular and extensible architecture

---

##  System Architecture (High Level)

IRDF consists of three main components:

1. **Backend (Python API)**  
   - Reinforcement learning engine (SAC)  
   - Feature-selection environment (state, action, reward logic)  
   - Evaluation pipelines (Decision Tree, Random Forest, Logistic Regression, SVM, KNN)  
   - REST endpoints for training, evaluation, and policy management  

2. **Frontend (Web UI)**  
   - Web interface for starting/stopping training and evaluation  
   - Real-time diagnostics: operation probabilities, feature probabilities, Q-values, entropy, F1-score, selected subset  
   - Views for inspecting saved policies and feature subsets  

3. **Optional Virtual IoT Testbed (Docker on Linux VM)**  
   - Used for latency and throughput experiments (as reported in the dissertation)  
   - Resource-constrained virtual IoT node running the trained detection model  

---

##  Installation (Windows 11 Host)

IRDF runs primarily on a **Windows 11** host system. The backend, frontend, and database run on Windows; the optional Docker-based testbed runs in a Linux virtual machine.

### 1. Clone the repository

```bash
git clone https://github.com/u04960174/IRDF-IoT-Ransomware-Detection-Framework.git
cd IRDF-IoT-Ransomware-Detection-Framework
