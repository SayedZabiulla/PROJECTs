const express = require("express");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const User = require("../models/User");

const router = express.Router();

// Signup
router.post("/signup", async (req, res) => {
  const { email, password, role } = req.body;

  const hashed = await bcrypt.hash(password, 10);

  const user = new User({ email, password: hashed, role });
  await user.save();

  res.json({ message: "User created" });
});

// Login
router.post("/login", async (req, res) => {
  const { email, password } = req.body;

  const user = await User.findOne({ email });

  if (!user) return res.status(400).json({ msg: "User not found" });

  const isMatch = await bcrypt.compare(password, user.password);

  if (!isMatch) return res.status(400).json({ msg: "Invalid password" });

  const token = jwt.sign({ id: user._id, role: user.role }, process.env.JWT_SECRET);

  res.json({ token, role: user.role });
});

module.exports = router;