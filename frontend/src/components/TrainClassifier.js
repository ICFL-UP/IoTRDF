import React from "react";

const TrainClassifier = ({ onNext }) => {
  return (
    <div>
      <h2>Train Classifier</h2>
      <p>Training the classifier with selected features...</p>
      <button onClick={onNext} style={{ marginTop: "10px" }}>
        Next: Test Classifier
      </button>
    </div>
  );
};

export default TrainClassifier;
