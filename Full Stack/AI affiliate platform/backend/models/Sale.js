const mongoose = require("mongoose");

const saleSchema = new mongoose.Schema({
  influencerId: mongoose.Schema.Types.ObjectId,
  amount: Number,
  date: { type: Date, default: Date.now }
});

module.exports = mongoose.model("Sale", saleSchema);