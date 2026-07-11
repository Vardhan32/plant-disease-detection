import React, { useState } from "react";
import FileUpload from "../components/FileUpload";
import Result from "../components/Result";

function Home() {
  const [result, setResult] = useState(null);

  return (
    <div>
      <h1>Plant Disease Detection</h1>
      <FileUpload setResult={setResult} />
      {result && <Result result={result} />}
    </div>
  );
}

export default Home;
