const express = require("express");
const axios = require("axios");
const path = require("path");
require("dotenv").config();

const app = express();

const CLIENT_ID = process.env.GITHUB_CLIENT_ID || "YOUR_CLIENT_ID";
const CLIENT_SECRET = process.env.GITHUB_CLIENT_SECRET || "YOUR_CLIENT_SECRET";

// Serve static files (index.html, contributors.html, etc.)
app.use(express.static(path.join(__dirname)));

/* =======================
   🔐 GITHUB OAUTH
======================= */
app.get("/auth/github", (req, res) => {
  res.redirect(
    `https://github.com/login/oauth/authorize?client_id=${CLIENT_ID}&scope=read:user`
  );
});

app.get("/auth/github/callback", async (req, res) => {
  const code = req.query.code;

  try {
    const tokenRes = await axios.post(
      "https://github.com/login/oauth/access_token",
      {
        client_id: CLIENT_ID,
        client_secret: CLIENT_SECRET,
        code
      },
      { headers: { Accept: "application/json" } }
    );

    const token = tokenRes.data.access_token;

    // Fetch user info
    const userRes = await axios.get("https://api.github.com/user", {
      headers: { Authorization: `Bearer ${token}` }
    });

    res.json({ user: userRes.data, token });
  } catch (err) {
    res.status(500).json({ error: "OAuth failed", details: err.message });
  }
});

/* =======================
   📊 CONTRIBUTION PROXY (avoids rate limits)
======================= */
app.get("/api/commits", async (req, res) => {
  try {
    const r = await axios.get(
      "https://api.github.com/repos/GODLOCAL/godlocal/stats/commit_activity",
      { headers: { Accept: "application/vnd.github+json" } }
    );
    res.json(r.data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get("/api/contributors", async (req, res) => {
  try {
    const r = await axios.get(
      "https://api.github.com/repos/GODLOCAL/godlocal/contributors",
      { headers: { Accept: "application/vnd.github+json" } }
    );
    res.json(r.data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`GODLOCAL server running on :${PORT}`));
