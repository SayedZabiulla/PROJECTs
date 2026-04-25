const express = require("express");
const Influencer = require("../models/Influencer");

const router = express.Router();

// Create influencer
router.post("/create", async (req, res) => {
  const { userId } = req.body;

  const code = "INF" + Math.floor(Math.random() * 10000);

  const influencer = new Influencer({
    userId,
    referralCode: code
  });

  await influencer.save();

  res.json(influencer);
});

module.exports = router;