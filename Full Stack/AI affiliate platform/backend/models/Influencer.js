const mongoose = require("mongoose");

const influencerSchema = new mongoose.Schema({
  userId: mongoose.Schema.Types.ObjectId,
  referralCode: String,
  clicks: { type: Number, default: 0 },
  conversions: { type: Number, default: 0 },
  revenue: { type: Number, default: 0 }
});

module.exports = mongoose.model("Influencer", influencerSchema);