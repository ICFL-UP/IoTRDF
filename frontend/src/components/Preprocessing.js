import React, { useState } from "react";

const Preprocessing = ({ datasetId, setLogs }) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [preprocessingComplete, setPreprocessingComplete] = useState(false);
  const [preprocessingResults, setPreprocessingResults] = useState(null);

  const handleStartPreprocessing = () => {
    setIsProcessing(true);
    setErrorMessage("");

    fetch("http://localhost:5000/preprocess", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_id: datasetId }),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to preprocess the dataset.");
        }
        return response.json();
      })
      .then((data) => {
        setPreprocessingComplete(true);
        setPreprocessingResults(data);
        setLogs((prevLogs) => [...prevLogs, `Preprocessing completed for dataset ${datasetId}.`]);
      })
      .catch((error) => {
        setErrorMessage(error.message);
        setLogs((prevLogs) => [...prevLogs, `Error: ${error.message}`]);
      })
      .finally(() => {
        setIsProcessing(false);
      });
  };

  return (
    <div>
      <h2>Preprocessing</h2>
      <button
        onClick={handleStartPreprocessing}
        disabled={isProcessing || preprocessingComplete}
        style={{
          padding: "10px 20px",
          marginTop: "10px",
          backgroundColor: isProcessing ? "#6c757d" : "#28a745",
          color: "#fff",
          border: "none",
          borderRadius: "5px",
          cursor: isProcessing ? "not-allowed" : "pointer",
        }}
      >
        {isProcessing ? "Processing..." : "Start"}
      </button>
      {preprocessingResults && (
        <pre style={{ marginTop: "20px", backgroundColor: "#f4f4f4", padding: "10px", borderRadius: "5px" }}>
          {JSON.stringify(preprocessingResults, null, 2)}
        </pre>
      )}
      {errorMessage && <p style={{ color: "red" }}>{errorMessage}</p>}
    </div>
  );
};

export default Preprocessing;
