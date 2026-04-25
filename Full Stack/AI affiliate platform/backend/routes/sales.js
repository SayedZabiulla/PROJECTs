const express = require("express");
const Sale = require("../models/Sale");
const Influencer = require("../models/Influencer");
const Payment = require("../models/Payment");

const router = express.Router();

// Add sale using referral code
router.post("/add", async (req, res) => {
  try {
    const { referralCode, amount } = req.body;

    const influencer = await Influencer.findOne({ referralCode });

    if (!influencer) {
      return res.status(404).json({ message: "Invalid referral code" });
    }

    // Create sale
    const sale = new Sale({
      influencerId: influencer._id,
      amount
    });
    await sale.save();

    // Update influencer stats
    influencer.conversions += 1;
    influencer.revenue += amount;
    await influencer.save();

    // 💰 Commission calculation (10%)
    const commission = amount * 0.1;

    // Create payment (pending)
    const payment = new Payment({
      influencerId: influencer._id,
      amount: commission,
      status: "pending"
    });

    await payment.save();

    res.json({
      message: "Sale + Commission recorded ✅",
      commission
    });

  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;