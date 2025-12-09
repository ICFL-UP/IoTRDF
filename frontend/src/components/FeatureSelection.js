import React from "react";

const FeatureSelection = ({ onNext }) => {
  return (
    <div>
      <h2>Performing Feature Selection</h2>
      <p>Using reinforcement learning to select the most relevant features...</p>
      <button onClick={onNext} style={{ marginTop: "10px" }}>
        Next: Train Classifier
      </button>
    </div>
  );
};

export default FeatureSelection;
