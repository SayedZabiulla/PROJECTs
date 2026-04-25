const mongoose = require("mongoose");

const paymentSchema = new mongoose.Schema({
  influencerId: mongoose.Schema.Types.ObjectId,
  amount: Number,
  status: {
    type: String,
    enum: ["pending", "approved", "paid"],
    default: "pending"
  }
});

module.exports = mongoose.model("Payment", paymentSchema);