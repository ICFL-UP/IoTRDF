import React from "react";

const TestClassifier = ({ onNext }) => {
  return (
    <div>
      <h2>Test Classifier</h2>
      <p>Evaluating the classifier on test data...</p>
      <button onClick={onNext} style={{ marginTop: "10px" }}>
        Next: Final Evaluation
      </button>
    </div>
  );
};

export default TestClassifier;
