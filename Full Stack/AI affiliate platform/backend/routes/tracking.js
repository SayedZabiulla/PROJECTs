const express = require("express");
const Influencer = require("../models/Influencer");

const router = express.Router();

// 🔗 Track clicks using referral code
router.get("/track", async (req, res) => {
  try {
    const { ref } = req.query;

    const influencer = await Influencer.findOne({ referralCode: ref });

    if (!influencer) {
      return res.status(404).json({ message: "Invalid referral code" });
    }

    influencer.clicks += 1;
    await influencer.save();

    res.json({
      message: "Click tracked ✅",
      influencerId: influencer._id
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;