import React from "react";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";

function App() {
  const token = localStorage.getItem("token");

  return (
    <div>
      {token ? <Dashboard /> : <Login />}
    </div>
  );
}

export default App;