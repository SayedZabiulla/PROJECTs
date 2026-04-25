const express = require("express");
const Influencer = require("../models/Influencer");
const Sale = require("../models/Sale");

const router = express.Router();

router.get("/", async (req, res) => {
  const influencers = await Influencer.find();
  const sales = await Sale.find();

  const totalRevenue = sales.reduce((acc, s) => acc + s.amount, 0);

  res.json({
    totalRevenue,
    influencers,
    totalSales: sales.length
  });
});

module.exports = router;