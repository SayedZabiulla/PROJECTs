const express = require("express");
const Influencer = require("../models/Influencer");
const Sale = require("../models/Sale");

const router = express.Router();

router.get("/insights", async (req, res) => {
  try {
    const influencers = await Influencer.find();
    const sales = await Sale.find();

    let insights = [];

    // 🔥 Insight 1: Top influencer
    if (influencers.length > 0) {
      const top = influencers.reduce((a, b) =>
        a.revenue > b.revenue ? a : b
      );
      insights.push(`Top influencer is ${top.referralCode} with ₹${top.revenue}`);
    }

    // 🔥 Insight 2: Low conversion detection
    influencers.forEach((inf) => {
      if (inf.clicks > 10 && inf.conversions < 2) {
        insights.push(
          `${inf.referralCode} has high clicks but low conversions`
        );
      }
    });

    // 🔥 Insight 3: Total revenue insight
    const totalRevenue = sales.reduce((sum, s) => sum + s.amount, 0);

    if (totalRevenue > 0) {
      insights.push(`Total revenue generated is ₹${totalRevenue}`);
    }

    res.json({ insights });

  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;