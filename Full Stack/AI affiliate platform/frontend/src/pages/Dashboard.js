import React, { useEffect, useState } from "react";
import API from "../api";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid,
  BarChart, Bar,
  PieChart, Pie, Cell
} from "recharts";

function Dashboard() {
  const [data, setData] = useState(null);
  const [insights, setInsights] = useState([]);

  useEffect(() => {
    // Get dashboard data
    API.get("/dashboard").then((res) => {
      setData(res.data);
    });

    // Get AI insights
    API.get("/ai/insights").then((res) => {
      setInsights(res.data.insights);
    });

  }, []);

  if (!data) return <h2>Loading...</h2>;

  // 📈 Line Chart Data
  const salesData = [
    { name: "Sales", value: data.totalSales }
  ];

  // 📊 Bar Chart
  const influencerData = data.influencers.map((inf) => ({
    name: inf.referralCode,
    revenue: inf.revenue
  }));

  // 🥧 Pie Chart
  const pieData = data.influencers.map((inf) => ({
    name: inf.referralCode,
    value: inf.revenue
  }));

  const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042"];

  return (
    <div style={{ padding: 20 }}>
      <h2>📊 Dashboard</h2>

      <p><b>Total Revenue:</b> ₹{data.totalRevenue}</p>
      <p><b>Total Sales:</b> {data.totalSales}</p>

      {/* 📈 LINE CHART */}
      <h3>Sales Overview</h3>
      <LineChart width={400} height={250} data={salesData}>
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <CartesianGrid stroke="#ccc" />
        <Line type="monotone" dataKey="value" stroke="#8884d8" />
      </LineChart>

      {/* 📊 BAR CHART */}
      <h3>Top Influencers</h3>
      <BarChart width={400} height={250} data={influencerData}>
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Bar dataKey="revenue" fill="#82ca9d" />
      </BarChart>

      {/* 🥧 PIE CHART */}
      <h3>Revenue Split</h3>
      <PieChart width={400} height={300}>
        <Pie
          data={pieData}
          dataKey="value"
          cx="50%"
          cy="50%"
          outerRadius={100}
          label
        >
          {pieData.map((entry, index) => (
            <Cell key={index} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip />
      </PieChart>

      {/* 🤖 AI INSIGHTS */}
      <h3>🤖 AI Insights</h3>
      <ul>
        {insights.map((ins, i) => (
          <li key={i}>{ins}</li>
        ))}
      </ul>

    </div>
  );
}

export default Dashboard;