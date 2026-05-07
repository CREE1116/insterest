import React, { createContext, useContext, useState, useEffect } from 'react';
import client from '../api/client';

interface User {
  id: string;
  user_id?: string;
  email: string;
  nickname?: string;
  profile_image?: string;
  bio?: string;
}

interface AuthContextType {
  user: User | null;
  isLoggedIn: boolean;
  login: (token: string) => Promise<void>;
  logout: () => void;
  checkAuth: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);

  const checkAuth = async () => {
    const token = localStorage.getItem('token');
    if (!token) {
      setUser(null);
      return;
    }

    // 요청 전에 헤더를 한 번 더 확실히 설정
    client.defaults.headers.common.Authorization = `Bearer ${token}`;
    
    try {
      const res = await client.get('/users/me');
      setUser(res.data);
    } catch (err) {
      console.error("Auth check failed:", err);
      setUser(null);
      localStorage.removeItem('token');
      delete client.defaults.headers.common.Authorization;
    }
  };

  const login = async (token: string) => {
    localStorage.setItem('token', token);
    client.defaults.headers.common.Authorization = `Bearer ${token}`;
    await checkAuth();
  };

  const logout = () => {
    localStorage.removeItem('token');
    delete client.defaults.headers.common.Authorization;
    setUser(null);
    client.post('/auth/logout');
  };

  useEffect(() => {
    checkAuth();
  }, []);

  return (
    <AuthContext.Provider value={{ user, isLoggedIn: !!user, login, logout, checkAuth }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used within AuthProvider');
  return context;
};
