import React from "react";

function Result({ result }) {
  return (
    <div>
      <h2>Prediction Result</h2>
      <p><strong>Disease:</strong> {result.class_name}</p>
      <p><strong>Confidence:</strong> {result.confidence}%</p>
      <h3>Precautions:</h3>
      <p>{result.precautions}</p>
    </div>
  );
}

export default Result;
