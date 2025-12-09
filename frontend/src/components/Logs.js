import React from "react";
import PropTypes from "prop-types";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faEye } from "@fortawesome/free-solid-svg-icons";

const Logs = ({ logs }) => {
  const listStyle = { 
    listStyleType: "none", 
    padding: 0,
    fontFamily: "'Roboto Mono', monospace",
    fontSize: "0.9rem"
  };

  const formatLog = (log) => {
    // Handle React elements first (including eye icon elements)
    if (React.isValidElement(log)) {
      return (
        <div style={{ 
          display: "flex",
          alignItems: "center",
          margin: "6px 0",
          padding: "4px 0",
          lineHeight: "1.4"
        }}>
          {log}
        </div>
      );
    }

    // Handle strings
    if (typeof log === "string") {
      if (log.includes("Preprocessing completed successfully.")) {
        return (
          <div style={{ 
            color: "#2E7D32", 
            fontWeight: "bold",
            margin: "8px 0",
            padding: "6px 0",
            borderBottom: "1px solid #e0e0e0"
          }}>
            {log}
          </div>
        );
      }
      if (log.includes("Shape")) {
        const isFeature = log.toLowerCase().includes("features");
        const isLabel = log.toLowerCase().includes("labels");
        const color = isFeature ? "#1565C0" : isLabel ? "#6A1B9A" : "#424242";
        return (
          <div style={{ 
            color,
            fontWeight: isFeature || isLabel ? "500" : "normal",
            margin: "4px 0"
          }}>
            {log}
          </div>
        );
      }
      return <div style={{ color: "#212121", margin: "4px 0" }}>{log}</div>;
    }

    // Handle objects with epoch data
    if (log.epoch) {
      const lossValue = typeof log.loss === "number" ? log.loss.toFixed(4) : "N/A";
      const bestLossValue = typeof log.best_loss === "number" ? log.best_loss.toFixed(4) : "N/A";
      const improvementValue = typeof log.improvement === "number" ? (log.improvement * 100).toFixed(2) : "N/A";
      const rolloutLength = typeof log.rollout_length === "number" ? log.rollout_length : "N/A";

      return (
        <div style={{ 
          marginBottom: "12px",
          padding: "8px",
          backgroundColor: "#f5f5f5",
          borderRadius: "4px"
        }}>
          <strong>📉 Epoch {log.epoch}:</strong>
          <ul style={listStyle}>
            <li>Loss: {lossValue}</li>
            <li>Best Loss: {bestLossValue}</li>
            <li>Improvement: {improvementValue}%</li>
            <li>Rollout Length: {rolloutLength}</li>
          </ul>

          {log.transitions && (
            <div style={{ marginTop: "10px" }}>
              <strong>📜 Transitions:</strong>
              <table style={{ 
                width: "100%", 
                borderCollapse: "collapse", 
                marginTop: "10px",
                fontSize: "0.85rem"
              }}>
                <thead>
                  <tr>
                    <th style={{ border: "1px solid #ddd", padding: "8px", backgroundColor: "#f0f0f0" }}>State</th>
                    <th style={{ border: "1px solid #ddd", padding: "8px", backgroundColor: "#f0f0f0" }}>Action</th>
                    <th style={{ border: "1px solid #ddd", padding: "8px", backgroundColor: "#f0f0f0" }}>Next State</th>
                    <th style={{ border: "1px solid #ddd", padding: "8px", backgroundColor: "#f0f0f0" }}>Reward</th>
                    <th style={{ border: "1px solid #ddd", padding: "8px", backgroundColor: "#f0f0f0" }}>Done</th>
                  </tr>
                </thead>
                <tbody>
                  {log.transitions.map((transition, index) => (
                    <tr key={index}>
                      <td style={{ border: "1px solid #ddd", padding: "8px", backgroundColor: "#fff" }}>
                        {JSON.stringify(transition.state)}
                      </td>
                      <td style={{ border: "1px solid #ddd", padding: "8px", backgroundColor: "#fff" }}>
                        {JSON.stringify(transition.action)}
                      </td>
                      <td style={{ border: "1px solid #ddd", padding: "8px", backgroundColor: "#fff" }}>
                        {JSON.stringify(transition.next_state)}
                      </td>
                      <td style={{ border: "1px solid #ddd", padding: "8px", backgroundColor: "#fff" }}>
                        {transition.reward}
                      </td>
                      <td style={{ border: "1px solid #ddd", padding: "8px", backgroundColor: "#fff" }}>
                        {transition.done ? "Yes" : "No"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      );
    }

    // Handle other object types
    if (typeof log === "object") {
      // Validation Metrics
      if (log.metrics) {
        return (
          <div style={{ 
            marginBottom: "12px",
            padding: "8px",
            backgroundColor: "#f5f5f5",
            borderRadius: "4px"
          }}>
            <strong>📊 Validation Metrics:</strong>
            <ul style={listStyle}>
              <li>MSE: {typeof log.metrics.mse === "number" ? log.metrics.mse.toFixed(4) : "N/A"}</li>
              <li>SPE: {typeof log.metrics.spe === "number" ? log.metrics.spe.toFixed(4) : "N/A"}</li>
              <li>NPE: {typeof log.metrics.npe === "number" ? log.metrics.npe.toFixed(4) : "N/A"}</li>
            </ul>
          </div>
        );
      }

      // Sample Predictions
      if (log.sample_predictions) {
        let samples = [];
        try {
          samples = Array.isArray(log.sample_predictions)
            ? log.sample_predictions
            : JSON.parse(log.sample_predictions);
        } catch (error) {
          console.error("Error parsing sample_predictions:", error);
        }

        return (
          <div style={{ 
            marginBottom: "12px",
            padding: "8px",
            backgroundColor: "#f5f5f5",
            borderRadius: "4px"
          }}>
            <strong> Sample Predictions:</strong>
            <ul style={listStyle}>
              {samples.map((sample, index) => (
                <li key={index} style={{ marginBottom: "10px" }}>
                  <strong>Sample {index + 1}:</strong>
                  <ul style={listStyle}>
                    <li>State: {JSON.stringify(sample.state)}</li>
                    <li>Predicted Next State: {sample.predicted_next_state}</li>
                    <li>Actual Next State: {sample.actual_next_state}</li>
                    <li>Error: {typeof sample.error === "number" ? sample.error.toFixed(6) : "N/A"}</li>
                  </ul>
                </li>
              ))}
            </ul>
          </div>
        );
      }

      // Model Predictions
      if (log.predicted_next_state) {
        return (
          <div style={{ 
            marginBottom: "12px",
            padding: "8px",
            backgroundColor: "#f5f5f5",
            borderRadius: "4px"
          }}>
            <strong>📈 Model Prediction:</strong>
            <ul style={listStyle}>
              <li>Mean Prediction: {typeof log.predicted_next_state === "number" ? log.predicted_next_state.toFixed(6) : "N/A"}</li>
              <li>Uncertainty: {typeof log.uncertainty === "number" ? log.uncertainty.toFixed(6) : "N/A"}</li>
            </ul>
          </div>
        );
      }

      // Messages
      if (log.message) {
        return (
          <div style={{ 
            color: "#2E7D32", 
            fontWeight: "bold",
            margin: "8px 0",
            padding: "4px 0"
          }}>
            {log.message}
          </div>
        );
      }

      // Errors
      if (log.error) {
        return (
          <div style={{ 
            color: "#c62828", 
            fontWeight: "bold",
            margin: "8px 0",
            padding: "4px 0"
          }}>
            {log.error}
          </div>
        );
      }

      // Model Summary
      if (log.model_summary) {
        const modelSummary = log.model_summary || {};
        const finalLossValue =
          typeof modelSummary.final_loss === "number" ? modelSummary.final_loss.toFixed(4) : "N/A";

        return (
          <div style={{ 
            marginBottom: "12px",
            padding: "12px",
            backgroundColor: "#f5f5f5",
            borderRadius: "4px",
            borderLeft: "4px solid #1565C0"
          }}>
            <strong>📝 Model Summary:</strong>
            <ul style={listStyle}>
              <li>🔢 Ensemble Size: {modelSummary.ensemble_size ?? "N/A"}</li>
              <li>📊 Input Shape: {JSON.stringify(modelSummary.input_shape) ?? "N/A"}</li>
              <li>⏳ Epochs Trained: {modelSummary.num_epochs_trained ?? "N/A"}</li>
              <li>📉 Final Loss: {finalLossValue}</li>
            </ul>
          </div>
        );
      }

      // Transitions
      if (log.transitions) {
        return (
          <div style={{ 
            marginBottom: "12px",
            padding: "8px",
            backgroundColor: "#f5f5f5",
            borderRadius: "4px"
          }}>
            <strong>📜 Transitions:</strong>
            <ul style={listStyle}>
              {log.transitions.map((transition, index) => (
                <li key={index} style={{ marginBottom: "10px" }}>
                  <strong>Transition #{index + 1}:</strong>
                  <ul style={listStyle}>
                    <li>State: {JSON.stringify(transition.state)}</li>
                    <li>Action: {JSON.stringify(transition.action)}</li>
                    <li>Next State: {JSON.stringify(transition.next_state)}</li>
                    <li>Reward: {transition.reward}</li>
                  </ul>
                </li>
              ))}
            </ul>
          </div>
        );
      }
    }

    // Fallback for any other type
    return <div style={{ color: "#616161", margin: "4px 0" }}>{String(log)}</div>;
  };

  return (
    <div
      style={{
        border: "1px solid #e0e0e0",
        padding: "16px",
        marginTop: "20px",
        height: "500px", // Increased from 300px to 500px
        overflowY: "auto",
        backgroundColor: "#fafafa",
        borderRadius: "8px",
        boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
        textAlign: "left",
      }}
    >
      {logs.length === 0 || (logs.length === 1 && logs[0] === "No output to show.") ? (
        <p style={{ 
          color: "#9e9e9e", 
          fontStyle: "italic",
          textAlign: "center",
          marginTop: "20px"
        }}>
          No output to show.
        </p>
      ) : (
        <ul style={listStyle}>
          {logs.map((log, index) => (
            <li key={index} style={{ 
              margin: "8px 0",
              padding: "4px 0",
              borderBottom: index < logs.length - 1 ? "1px solid #eee" : "none"
            }}>
              {formatLog(log)}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

Logs.propTypes = {
  logs: PropTypes.arrayOf(
    PropTypes.oneOfType([
      PropTypes.string,
      PropTypes.element,
      PropTypes.shape({
        epoch: PropTypes.number,
        loss: PropTypes.number,
        best_loss: PropTypes.number,
        improvement: PropTypes.number,
        rollout_length: PropTypes.number,
        metrics: PropTypes.shape({
          mse: PropTypes.number,
          spe: PropTypes.number,
          npe: PropTypes.number,
        }),
        sample_predictions: PropTypes.oneOfType([
          PropTypes.string,
          PropTypes.array,
        ]),
        predicted_next_state: PropTypes.number,
        uncertainty: PropTypes.number,
        message: PropTypes.string,
        error: PropTypes.string,
        model_summary: PropTypes.shape({
          ensemble_size: PropTypes.number,
          input_shape: PropTypes.array,
          num_epochs_trained: PropTypes.number,
          final_loss: PropTypes.number,
        }),
        transitions: PropTypes.array,
      }),
    ])
  ).isRequired,
};

export default Logs;