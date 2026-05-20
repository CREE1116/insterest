import axios from 'axios';

const getBaseURL = () => {
  // If we are accessing via localhost:30088 (Kubernetes exposed NodePort),
  // we want API requests to hit port 80 (Ingress gateway).
  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname;
    // Local dev or kind cluster
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      return 'http://localhost/api/v1';
    }
    // Remote or cluster IP
    return `${window.location.protocol}//${hostname}/api/v1`;
  }
  return '/api/v1';
};

const client = axios.create({
  baseURL: getBaseURL(),
  withCredentials: true,
});

// Request Interceptor to carry JWT if available
client.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
}, (error) => {
  return Promise.reject(error);
});

export default client;
