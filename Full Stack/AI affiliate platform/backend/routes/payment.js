const express = require("express");
const Payment = require("../models/Payment");

const router = express.Router();

// Get all payments
router.get("/", async (req, res) => {
  try {
    const payments = await Payment.find();
    res.json(payments);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Update payment status
router.put("/update/:id", async (req, res) => {
  try {
    const { status } = req.body;

    // ✅ Validate status
    if (!["pending", "approved", "paid"].includes(status)) {
      return res.status(400).json({ message: "Invalid status" });
    }

    const payment = await Payment.findByIdAndUpdate(
      req.params.id,
      { status },
      { new: true }
    );

    // ✅ If ID not found
    if (!payment) {
      return res.status(404).json({ message: "Payment not found" });
    }

    res.json(payment);

  } catch (err) {
    console.error("UPDATE ERROR:", err);   // 👈 IMPORTANT
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;