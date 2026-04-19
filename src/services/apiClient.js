import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Create axios instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor to handle token refresh
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    // If 401 and not already retried, try to refresh token
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      try {
        const refreshToken = localStorage.getItem('refresh_token');
        if (refreshToken) {
          const response = await axios.post(`${API_BASE_URL}/api/auth/refresh`, {
            refresh_token: refreshToken,
          });

          const { access_token, refresh_token } = response.data;
          localStorage.setItem('access_token', access_token);
          localStorage.setItem('refresh_token', refresh_token);

          originalRequest.headers.Authorization = `Bearer ${access_token}`;
          return apiClient(originalRequest);
        }
      } catch (refreshError) {
        // Refresh failed, logout user
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        window.location.href = '/login';
        return Promise.reject(refreshError);
      }
    }

    return Promise.reject(error);
  }
);

// API methods
const api = {
  // Auth
  auth: {
    register: (data) => apiClient.post('/api/auth/register', data),
    login: (data) => apiClient.post('/api/auth/login', data),
    me: () => apiClient.get('/api/auth/me'),
    refresh: (refreshToken) => apiClient.post('/api/auth/refresh', { refresh_token: refreshToken }),
  },

  // Devices
  devices: {
    getAll: (params) => apiClient.get('/api/devices', { params }),
    getById: (id) => apiClient.get(`/api/devices/${id}`),
    getBrands: () => apiClient.get('/api/devices/brands'),
    upload: (file) => {
      const formData = new FormData();
      formData.append('file', file);
      return apiClient.post('/api/devices/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
    },
    uploadFiles: (files, append = false) => {
      const formData = new FormData();
      if (Array.isArray(files)) {
        files.forEach((f) => formData.append('files', f));
      } else if (files instanceof FileList) {
        Array.from(files).forEach((f) => formData.append('files', f));
      } else {
        formData.append('files', files);
      }
      formData.append('append', append ? 'true' : 'false');
      return apiClient.post('/api/devices/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
    },
  },

  // User features
  user: {
    favorites: {
      getAll: () => apiClient.get('/api/user/favorites'),
      add: (deviceId, note = '') => apiClient.post('/api/user/favorites', { device_id: deviceId, note }),
      remove: (deviceId) => apiClient.delete(`/api/user/favorites/${deviceId}`),
    },
    searches: {
      getAll: () => apiClient.get('/api/user/searches'),
      create: (name, filters) => apiClient.post('/api/user/searches', { name, filters }),
      update: (id, data) => apiClient.put(`/api/user/searches/${id}`, data),
      delete: (id) => apiClient.delete(`/api/user/searches/${id}`),
    },
  },

  // AI Recommendations
  recommendations: {
    get: (data) => apiClient.post('/api/recommend', data),
    parse: (query) => apiClient.get('/api/recommend/parse', { params: { query } }),
  },

  // Price tracking
  priceTracking: {
    subscribe: (deviceId, targetPrice) => 
      apiClient.post('/api/price-track/subscribe', { device_id: deviceId, target_price: targetPrice }),
    getHistory: (deviceId, days = 30) => 
      apiClient.get(`/api/price-track/${deviceId}/history`, { params: { days } }),
    unsubscribe: (deviceId) => 
      apiClient.delete(`/api/price-track/unsubscribe/${deviceId}`),
  },

  // Comparisons
  comparisons: {
    getAll: () => apiClient.get('/api/compare'),
    create: (deviceIds, name = '') => apiClient.post('/api/compare', { device_ids: deviceIds, name }),
    getById: (id) => apiClient.get(`/api/compare/${id}`),
    delete: (id) => apiClient.delete(`/api/compare/${id}`),
  },
};

export default api;
export { apiClient };


