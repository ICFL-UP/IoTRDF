import React, { useState } from "react";

const TrainEnvironmentModel = ({ onNext, onBack, datasetId, workAreaMessage, setWorkAreaMessage, logs, setLogs }) => {
  const [isTraining, setIsTraining] = useState(false);
  const [isTrainingComplete, setIsTrainingComplete] = useState(false);

  const handleDynamicsModel = () => {
    setLogs((prevLogs) => [...prevLogs, "Training environment model started."]);
    setWorkAreaMessage("Training the environment model. Please wait...");

    fetch("http://localhost:5000/train_environment_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id: selectedDataset }),
    })
        .then((response) => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.error || "Training environment model failed.");
                });
            }
            return response.json();
        })
        .then((data) => {
            if (data.error) {
                throw new Error(data.error);
            }

            // Log the training process and final model details
            const trainingLogs = data.logs || [];
            const formattedLogs = trainingLogs.map((log) => `• ${log}`);

            setLogs((prevLogs) => [
                ...prevLogs,
                "Training environment model completed successfully.",
                ...formattedLogs,
                `Final Model: ${data.final_model || "Details not available"}`,
            ]);

            // Update the UI to allow validation
            setWorkAreaMessage("Environment model training is completed. Proceed to validation.");
            setNextStep("validate");  // Update the next step to "validate"
        })
        .catch((error) => {
            setWorkAreaMessage("Error occurred during training environment model. Try again.");
            setLogs((prevLogs) => [...prevLogs, `Error during training: ${error.message}`]);
        });
};

const handleValidateModel = () => {
    setLogs((prevLogs) => [...prevLogs, "Validation started."]);
    setWorkAreaMessage("Validation is in progress...");

    fetch("http://localhost:5000/validate_environment_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id: selectedDataset }),
    })
        .then((response) => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.error || "Validation failed. Please check the backend logs.");
                });
            }
            return response.json();
        })
        .then((data) => {
            if (data.error) {
                throw new Error(data.error);
            }

            // Log the validation process and results
            const validationLogs = data.logs || [];
            const formattedLogs = validationLogs.map((log) => `• ${log}`);

            setLogs((prevLogs) => [
                ...prevLogs,
                "Validation completed successfully.",
                ...formattedLogs,
                `Validation Summary: ${data.validation_summary || "Details not available"}`,
            ]);

            // Update the UI to indicate completion
            setWorkAreaMessage("Validation completed. Process finished.");
            setNextStep("complete");  // Update the next step to "complete"
        })
        .catch((error) => {
            setWorkAreaMessage("Error occurred during validation. Try again.");
            setLogs((prevLogs) => [...prevLogs, `Error during validation: ${error.message}`]);
        });
};


  return (
    <div>
      <p style={{ color: "red", fontWeight: "bold", marginBottom: "10px" }}>
        {workAreaMessage}
      </p>
      <div style={{ display: "flex", justifyContent: "center", gap: "20px" }}>
        <button
          onClick={onBack}
          style={{
            padding: "10px 20px",
            borderRadius: "5px",
            backgroundColor: "#FFC107",
            color: "#fff",
            border: "none",
          }}
        >
          Back
        </button>
        {isTrainingComplete ? (
          <button
            onClick={onNext}
            style={{
              padding: "10px 20px",
              borderRadius: "5px",
              backgroundColor: "#007BFF",
              color: "#fff",
              border: "none",
            }}
          >
            Continue
          </button>
        ) : (
          <button
            onClick={handleDynamicsModel}
            style={{
              padding: "10px 20px",
              borderRadius: "5px",
              backgroundColor: isTraining ? "#ccc" : "#28A745",
              color: "#fff",
              border: "none",
              cursor: isTraining ? "not-allowed" : "pointer",
            }}
            disabled={isTraining}
          >
            {isTraining ? "Training..." : "Train"}
          </button>
        )}
      </div>
    </div>
  );
};

export default TrainEnvironmentModel;
