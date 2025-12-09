import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  BarChart,
  Bar,
  Legend,
  ScatterChart,
  Scatter,
  ResponsiveContainer,
} from "recharts";

const VisualizationPanel = ({ visualizationData }) => {
  if (!visualizationData || Object.keys(visualizationData).length === 0) {
    return (
      <div
        style={{
          padding: "10px",
          marginTop: "20px",
          height: "200px",
          overflowY: "auto",
          backgroundColor: "#ffffff",
          border: "1px solid #ddd",
          borderRadius: "5px",
          boxShadow: "0px 2px 5px rgba(0, 0, 0, 0.1)",
          textAlign: "center",
        }}
      >
        <p style={{ fontStyle: "italic", color: "#aaa" }}>
          No visualization data available yet.
        </p>
      </div>
    );
  }

  return (
    <div
      style={{
        padding: "10px",
        marginTop: "20px",
        backgroundColor: "#ffffff",
        border: "1px solid #ddd",
        borderRadius: "5px",
        boxShadow: "0px 2px 5px rgba(0, 0, 0, 0.1)",
      }}
    >
      <h3>Visualization Panel</h3>

      {/* Training Loss Curve */}
      {visualizationData.trainingLoss && (
        <>
          <h4>Training Loss Over Time</h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={visualizationData.trainingLoss}>
              <XAxis dataKey="epoch" name="Epoch" />
              <YAxis name="Loss" />
              <Tooltip />
              <CartesianGrid stroke="#eee" strokeDasharray="5 5" />
              <Line
                type="monotone"
                dataKey="loss"
                stroke="#007bff"
                strokeWidth={2}
                dot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </>
      )}

      {/* Prediction Error Histogram */}
      {visualizationData.predictionErrors && (
        <>
          <h4>Prediction Error Distribution</h4>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={visualizationData.predictionErrors}>
              <XAxis dataKey="error" name="Error" />
              <YAxis name="Count" />
              <Tooltip />
              <Legend />
              <CartesianGrid strokeDasharray="3 3" />
              <Bar dataKey="count" fill="#ff7300" />
            </BarChart>
          </ResponsiveContainer>
        </>
      )}

      {/* Actual vs. Predicted Next State Scatter Plot */}
      {visualizationData.nextStateComparison && (
        <>
          <h4>Actual vs. Predicted Next State</h4>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart>
              <CartesianGrid />
              <XAxis type="number" dataKey="actual" name="Actual" />
              <YAxis type="number" dataKey="predicted" name="Predicted" />
              <Tooltip cursor={{ strokeDasharray: "3 3" }} />
              <Scatter
                name="Next State"
                data={visualizationData.nextStateComparison}
                fill="#82ca9d"
                line
              />
            </ScatterChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  );
};

export default VisualizationPanel;