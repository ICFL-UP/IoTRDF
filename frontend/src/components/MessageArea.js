import React from "react";

const MessageArea = ({ message }) => {
  return (
    <div
      style={{
        margin: "20px auto",
        padding: "10px",
        border: "1px solid #ccc",
        borderRadius: "5px",
        backgroundColor: "#f9f9f9",
        width: "80%",
        textAlign: "center",
        color: "#333",
      }}
    >
      <p>{message}</p>
    </div>
  );
};

export default MessageArea;
