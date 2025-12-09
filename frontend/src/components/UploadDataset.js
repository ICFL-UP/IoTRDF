import React, { useState, useEffect } from "react";

const UploadDataset = ({ onNext, setLogs }) => {
  const [datasets, setDatasets] = useState([]); // Existing datasets
  const [selectedDataset, setSelectedDataset] = useState(""); // Selected dataset
  const [uploadedFile, setUploadedFile] = useState(null); // Uploaded file
  const [uploadSuccess, setUploadSuccess] = useState(false); // Upload success status

  useEffect(() => {
    // Fetch existing datasets on component load
    fetch("http://localhost:5000/datasets")
      .then((response) => response.json())
      .then((data) => {
        setDatasets(data.datasets || []);
        setLogs((prevLogs) => [...prevLogs, "Fetched existing datasets successfully."]);
      })
      .catch((error) => {
        setLogs((prevLogs) => [...prevLogs, `Error fetching datasets: ${error.message}`]);
      });
  }, [setLogs]);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && (file.type === "text/csv" || file.type.includes("excel"))) {
      setUploadedFile(file);
      setLogs((prevLogs) => [...prevLogs, `File ${file.name} selected but not yet uploaded.`]);
    } else {
      setUploadedFile(null);
      setLogs((prevLogs) => [...prevLogs, "Invalid file type. Please upload a valid CSV or Excel file."]);
    }
  };

  const handleSubmit = () => {
    if (selectedDataset) {
      setLogs((prevLogs) => [...prevLogs, `Selected existing dataset ID: ${selectedDataset}.`]);
      setUploadSuccess(true);
    } else if (uploadedFile) {
      const formData = new FormData();
      formData.append("file", uploadedFile);

      fetch("http://localhost:5000/upload", {
        method: "POST",
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          setLogs((prevLogs) => [
            ...prevLogs,
            `File ${uploadedFile.name} uploaded successfully. Dataset ID: ${data.dataset_id}.`,
          ]);
          setUploadSuccess(true);
        })
        .catch((error) => {
          setLogs((prevLogs) => [...prevLogs, `Error uploading file: ${error.message}`]);
        });
    } else {
      setLogs((prevLogs) => [...prevLogs, "No file or dataset selected. Please try again."]);
    }
  };

  return (
    <div>
      <h2>Upload IoT Dataset</h2>
      <p style={{ color: "red" }}>Upload or select your IoT dataset to proceed:</p>
      {/* Dropdown to select existing dataset */}
      <div>
        <label htmlFor="datasetSelect" style={{ marginRight: "10px" }}>
          Select Existing Dataset:
        </label>
        <select
          id="datasetSelect"
          value={selectedDataset}
          onChange={(e) => setSelectedDataset(e.target.value)}
        >
          <option value="">--Select--</option>
          {datasets.map((dataset) => (
            <option key={dataset.id} value={dataset.id}>
              {dataset.filename}
            </option>
          ))}
        </select>
      </div>
      {/* File upload */}
      <div>
        <label htmlFor="fileInput" style={{ marginRight: "10px" }}>
          Choose File:
        </label>
        <input id="fileInput" type="file" accept=".csv, .xlsx, .xls" onChange={handleFileChange} />
      </div>
      {/* Submit button */}
      <div>
        <button
          style={{
            marginTop: "10px",
            padding: "10px 20px",
            backgroundColor: "#007BFF",
            color: "#fff",
            border: "none",
            borderRadius: "5px",
            cursor: "pointer",
          }}
          onClick={handleSubmit}
        >
          Submit
        </button>
      </div>
    </div>
  );
};

export default UploadDataset;
