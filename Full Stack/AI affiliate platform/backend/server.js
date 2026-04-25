const express = require("express");
const cors = require("cors");
const mongoose = require("mongoose");
require("dotenv").config();

const app = express();

app.use(cors());
app.use(express.json());

// Routes
app.use("/api/auth", require("./routes/auth"));
app.use("/api/influencer", require("./routes/influencer"));
app.use("/api/sales", require("./routes/sales"));
app.use("/api/dashboard", require("./routes/dashboard"));
app.use("/api", require("./routes/tracking"));
app.use("/api/payment", require("./routes/payment"));
app.use("/api/ai", require("./routes/ai"));

// Test route
app.get("/", (req, res) => {
  res.send("Backend is running 🚀");
});

const PORT = process.env.PORT || 5000;

// MongoDB connection + server start
mongoose.connect(process.env.MONGO_URI)
.then(() => {
  console.log("MongoDB Connected ✅");

  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
  });

})
.catch(err => console.log("MongoDB Error:", err));

// Optional error handler
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).send("Something broke!");
});