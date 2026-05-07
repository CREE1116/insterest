import axios from 'axios';

const client = axios.create({
  baseURL: '/api/v1',
  withCredentials: true,
});

// 모든 요청 시 최신 토큰을 동적으로 가져오도록 인터셉터 강화
client.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
}, (error) => {
  return Promise.reject(error);
});

// 401 에러 발생 시 로그아웃 처리 등을 위한 응답 인터셉터 (선택 사항)
client.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      // 만약 인증 에러가 나면 로컬 토큰 삭제
      localStorage.removeItem('token');
      // 필요 시 페이지 새로고침이나 로그인 모달 유도
    }
    return Promise.reject(error);
  }
);

export default client;
