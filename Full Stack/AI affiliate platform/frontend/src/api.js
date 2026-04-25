import axios from "axios";

const API = axios.create({
  baseURL: "https://ai-affiliate-backend.onrender.com/api"
});

export default API;