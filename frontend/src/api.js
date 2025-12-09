import axios from "axios";

const API_URL = "http://localhost:5000"; // Adjust if your backend runs on another port.

export const uploadDataset = (formData) =>
  axios.post(`${API_URL}/upload-dataset`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

export const startProcess = (processName) =>
  axios.post(`${API_URL}/process`, { process: processName });

export const fetchLogs = () => axios.get(`${API_URL}/logs`);
